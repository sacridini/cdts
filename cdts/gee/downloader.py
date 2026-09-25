import ee
import os
import math
import random
import time
import uuid
import threading
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

from .drive_sync import submit_drive_export, wait_and_download_task

# Earth Engine interactive-request limits (computePixels / getDownloadURL):
# https://developers.google.com/earth-engine/reference/rest/v1/projects.image/computePixels
_EE_MAX_REQUEST_MB = 32
_EE_MAX_TILE_DIM = 10000
_EE_MAX_TILE_BANDS = 1024

# EE converts a metric `scale` to degrees at the equator when the output CRS is
# geographic -- the same constant it uses for getDownloadURL(scale=..., crs='EPSG:4326').
_METERS_PER_DEGREE = 111319.49079327357

# Output GeoTIFF block size. Tile columns follow it so writes land on whole
# blocks (partial-block writes to a compressed GTiff force read-modify-write).
_BLOCK = 512
# Spatial tile granularity: divides _BLOCK, so tile edges never split a block
# into more than two writes.
_TILE_STEP = 256
# Smallest tile worth a request of its own (as in geedim's default). Each EE
# request pays a fixed ~2-4 s overhead, so smaller tiles only add latency --
# measured: a 13 MB NDVI image cut into 0.5 MB tiles took 2x longer.
_MIN_TILE_MB = 4

# Masked pixels come back from EE as these values (float -> -inf, signed int ->
# dtype min, unsigned -> 0); the output GeoTIFF's nodata tag is set to match.
_NODATA = {
    'uint8': 0, 'uint16': 0, 'uint32': 0,
    'int8': np.iinfo('int8').min, 'int16': np.iinfo('int16').min, 'int32': np.iinfo('int32').min,
    'float32': float('-inf'), 'float64': float('-inf'),
}
_CAST = {
    'uint8': ee.Image.toUint8, 'uint16': ee.Image.toUint16, 'uint32': ee.Image.toUint32,
    'int8': ee.Image.toInt8, 'int16': ee.Image.toInt16, 'int32': ee.Image.toInt32,
    'float32': ee.Image.toFloat, 'float64': ee.Image.toDouble,
}

# How a failed tile request is handled:
# - throttle: EE rejected it for concurrency/rate (HTTP 429) -> lower the
#   concurrency limit and retry soon;
# - transient: network / 5xx -> retry with exponential backoff;
# - compute_limit: the tile's computation is too heavy for the interactive API
#   (retrying is pointless; the Drive batch export has much higher limits);
# - fatal: anything else (bad expression, missing band...) -> give up.
_THROTTLE_MARKERS = ('too many requests', 'too many concurrent', 'concurrency limit', 'rate limit',
                     'quota exceeded', 'resource_exhausted', '429')
_TRANSIENT_MARKERS = ('internal error', 'service unavailable', 'backend error', 'capacity exceeded',
                      'deadline', 'bad gateway', 'gateway timeout')
_COMPUTE_LIMIT_MARKERS = ('memory limit', 'computation timed out', 'too many pixels',
                          'request payload size', 'total request size', 'too much memory')


class _ComputeLimitError(RuntimeError):
    """A tile hit an EE interactive compute limit -- direct download can't succeed."""


@dataclass(frozen=True)
class _Grid:
    crs: str
    transform: Affine
    width: int
    height: int


def _pixel_grid(bounds: Tuple[float, float, float, float], scale: float, crs: str,
                geographic: bool) -> _Grid:
    """Single pixel grid covering `bounds` (minx, miny, maxx, maxy in `crs` units),
    snapped to integer multiples of the resolution so separate downloads of
    adjacent ROIs share the same grid."""
    res = scale / _METERS_PER_DEGREE if geographic else float(scale)
    minx, miny, maxx, maxy = bounds
    eps = 1e-6  # in pixels: bounds already on the grid must not grow by a float-noise pixel
    x0 = math.floor(minx / res + eps) * res
    y0 = math.ceil(maxy / res - eps) * res
    width = max(1, math.ceil((maxx - x0) / res - eps))
    height = max(1, math.ceil((y0 - miny) / res - eps))
    return _Grid(crs, Affine(res, 0.0, x0, 0.0, -res, y0), width, height)


def _image_dtype(band_types: list) -> str:
    """Smallest dtype able to hold every band (EE PixelType dicts), matching
    what EE itself would promote a multi-band download to."""
    precisions = {t['precision'] for t in band_types}
    if 'double' in precisions:
        return 'float64'
    int_types = [t for t in band_types if t['precision'] == 'int']
    int_dtype = None
    if int_types:
        lo = min(t.get('min', np.iinfo('int32').min) for t in int_types)
        hi = max(t.get('max', np.iinfo('int32').max) for t in int_types)
        int_dtype = next((d for d in ('uint8', 'uint16', 'uint32', 'int8', 'int16', 'int32')
                          if np.iinfo(d).min <= lo and hi <= np.iinfo(d).max), 'float64')
    if 'float' in precisions:
        return str(np.promote_types('float32', int_dtype)) if int_dtype else 'float32'
    return int_dtype or 'float32'


def _tile_shape(n_bands: int, height: int, width: int, dtype: str, max_tile_mb: float,
                max_dim: int = _EE_MAX_TILE_DIM, max_bands: int = _EE_MAX_TILE_BANDS) -> Tuple[int, int, int]:
    """(bands, rows, cols) of the largest tile under `max_tile_mb` raw bytes.

    Splits space first (rows, then cols, in multiples of _TILE_STEP) and bands
    only once a _TILE_STEP x _TILE_STEP window of all bands still doesn't fit.
    Unlike geedim (which splits bands first), this avoids re-running an
    expensive per-pixel computation such as a medoid composite once per band
    slice over the same area -- every band of a composite is produced by the
    same computation, so fetching them in two requests doubles the EE work."""
    itemsize = np.dtype(dtype).itemsize
    budget = max(1, int(max_tile_mb * 2 ** 20 // itemsize))  # bands*pixels per tile
    capped_dim = max_dim // _BLOCK * _BLOCK or max_dim
    b = min(n_bands, max_bands)
    h = height if height <= max_dim else capped_dim
    w = width if width <= max_dim else capped_dim

    def shrink(dim: int, other: int, full: int) -> int:
        # largest multiple of _TILE_STEP (floored at _TILE_STEP, or the full
        # extent when smaller) keeping b*dim*other within budget
        fit = budget // (b * other)
        return max(min(full, _TILE_STEP), fit // _TILE_STEP * _TILE_STEP)

    if b * h * w > budget:
        h = min(h, shrink(h, w, height))
    if b * h * w > budget:
        w = min(w, shrink(w, h, width))
    if b * h * w > budget:
        b = max(1, budget // (h * w))
    # only reachable with a budget below one _TILE_STEP window of one band
    if b * h * w > budget:
        h = max(1, budget // w)
    if b * h * w > budget:
        w = max(1, budget // h)
    return b, h, w


class _GuidedScheduler:
    """Hands out tiles on demand, each sized to (remaining work / current
    concurrency) -- OpenMP's 'guided' schedule -- capped at `max_tile_mb`.

    Every EE request pays a fixed overhead (~2 s for a medoid composite) on
    top of compute proportional to its size, and the account's concurrency
    limit is unknown up front (~40 on a standard tier, 2-3 in Restricted
    Mode). A fixed tile plan is therefore wrong for one of the two: small
    tiles waste overhead when only 2 run at once; large tiles leave workers
    idle when 40 could. Sizing each tile when it is handed out lets the plan
    follow whatever concurrency _AdaptiveLimiter has discovered so far, and
    tiles shrink towards the end (down to _MIN_TILE_MB) so the last wave
    finishes together.

    The image is cut into lanes (band slices from _tile_shape x column
    stripes one block wide); tiles are runs of rows within a lane, in
    multiples of _TILE_STEP."""

    def __init__(self, n_bands: int, height: int, width: int, dtype: str, max_tile_mb: float):
        tb, _, tw = _tile_shape(n_bands, height, width, dtype, max_tile_mb)
        self.itemsize = np.dtype(dtype).itemsize
        self.height = height
        # lanes one GeoTIFF block wide: narrow enough that the smallest tile
        # (_TILE_STEP rows) stays small when concurrency is high, and every
        # write covers whole block columns
        lw = min(tw, _BLOCK)
        budget = int(max_tile_mb * 2 ** 20 // self.itemsize)
        fit = budget // (tb * lw)
        self.max_rows = max(1, min(height, _EE_MAX_TILE_DIM,
                                   fit // _TILE_STEP * _TILE_STEP if fit >= _TILE_STEP else fit))
        self.lanes = [[b0, min(b0 + tb, n_bands), c0, min(c0 + lw, width), 0]
                      for b0 in range(0, n_bands, tb) for c0 in range(0, width, lw)]
        self.total_bytes = self.remaining_bytes = n_bands * height * width * self.itemsize
        self._lock = threading.Lock()

    def next(self, concurrency: int) -> Optional[Tuple[int, int, Window]]:
        """(band_start, band_stop, window) of the next tile, or None when done."""
        with self._lock:
            open_lanes = [lane for lane in self.lanes if lane[4] < self.height]
            if not open_lanes:
                return None
            # the lane with the most work left, so lanes finish together
            lane = max(open_lanes, key=lambda l: (l[1] - l[0]) * (l[3] - l[2]) * (self.height - l[4]))
            b0, b1, c0, c1, r0 = lane
            left = self.height - r0
            row_bytes = (b1 - b0) * (c1 - c0) * self.itemsize
            target = int(self.remaining_bytes / max(1, concurrency) / row_bytes) // _TILE_STEP * _TILE_STEP
            min_rows = math.ceil(_MIN_TILE_MB * 2 ** 20 / row_bytes / _TILE_STEP) * _TILE_STEP
            rows = max(1, min(max(target, min_rows), self.max_rows, left))
            if left - rows < min_rows and left <= self.max_rows:
                rows = left  # don't leave a sliver for a request of its own
            lane[4] += rows
            self.remaining_bytes -= rows * row_bytes
            return b0, b1, Window(c0, r0, c1 - c0, rows)


def _classify_error(e: Exception) -> str:
    msg = str(e).lower()
    if any(m in msg for m in _COMPUTE_LIMIT_MARKERS):
        return 'compute_limit'
    if any(m in msg for m in _THROTTLE_MARKERS):
        return 'throttle'
    if not isinstance(e, ee.EEException) or any(m in msg for m in _TRANSIENT_MARKERS):
        return 'transient'  # network / IO errors and EE server errors
    return 'fatal'


class _AdaptiveLimiter:
    """Concurrency limit that adapts to EE's per-account interactive limit,
    which varies by account tier (a project in noncommercial Restricted Mode
    gets only 2-3 concurrent requests). TCP-style: starts at `initial` and
    grows by one per success (doubling each round) until the first HTTP 429,
    then backs off on each 429 (see _cut) and grows by one per `limit`
    consecutive successes."""

    def __init__(self, max_limit: int, initial: int = 4):
        self.max_limit = max(1, max_limit)
        self.limit = min(self.max_limit, max(1, initial))
        self._slow_start = True
        self._sustained = self.limit
        self._active = 0
        self._streak = 0
        self._epoch = 0  # bumped on every cut
        self._cond = threading.Condition()

    @staticmethod
    def _cut(limit: int) -> int:
        # Halve large limits, but step small ones down by one: EE's concurrency
        # accounting lags (a 429 can arrive with only one request in flight),
        # and the costs are lopsided -- a rejected request costs ~1 s, while an
        # idle slot costs a whole tile's compute time (~10-30 s).
        return limit // 2 if limit > 8 else limit - 1

    @property
    def target(self) -> int:
        """Concurrency to plan tile sizes for: the ceiling until EE first
        pushes back, then the limit right after the latest cut -- a
        conservative estimate of what the account sustains, which ignores
        the additive probing above it that EE will cut again."""
        return self.max_limit if self._slow_start else self._sustained

    def acquire(self) -> int:
        """Wait for a slot; returns a token to pass back to release()."""
        with self._cond:
            while self._active >= self.limit:
                self._cond.wait()
            self._active += 1
            return self._epoch

    def release(self, token: int, throttled: bool = False):
        with self._cond:
            self._active -= 1
            if throttled:
                self._streak = 0
                self._slow_start = False
                # only requests started under the current limit can cut it --
                # the rest of a burst sent before the last cut is stale signal
                if token == self._epoch:
                    self.limit = self._sustained = max(1, self._cut(self.limit))
                    self._epoch += 1
            elif self.limit < self.max_limit:
                self._streak += 1
                if self._slow_start or self._streak >= self.limit:
                    self.limit += 1
                    self._streak = 0
            self._cond.notify_all()


def _compute_pixels(request: dict) -> bytes:
    """ee.data.computePixels, but without the client library's own silent
    retries of HTTP 429 (up to ~30 s of sleeping while holding a concurrency
    slot) so throttling reaches _AdaptiveLimiter immediately. Falls back to
    the public call if the private helpers it relies on ever move."""
    try:
        body = dict(request, expression=ee.serializer.encode(request['expression']))
        call = ee.data._get_cloud_projects_raw().image().computePixels(
            project=ee.data._get_projects_path(), body=body)
    except AttributeError:
        return ee.data.computePixels(request)
    return ee.data._execute_cloud_call(call, num_retries=0)


def _fetch_tile(image: ee.Image, band_ids: list, grid: _Grid, window: Window, limiter: _AdaptiveLimiter,
                token: Optional[int], max_retries: int, base_backoff: float) -> np.ndarray:
    """One computePixels call (a single round trip, vs. getDownloadURL's two)
    for `band_ids` over `window` of `grid`, decoded to a (bands, rows, cols)
    array. `token` is the limiter slot already held for the first attempt
    (None to acquire one); retries acquire their own. Throttled requests
    shrink the concurrency limit and are retried after a short jittered
    pause; transient failures are retried with exponential backoff; raises
    _ComputeLimitError immediately when retrying cannot help."""
    x = grid.transform.c + window.col_off * grid.transform.a
    y = grid.transform.f + window.row_off * grid.transform.e
    request = {
        'expression': image,
        'fileFormat': 'GEO_TIFF',
        'bandIds': band_ids,
        'grid': {
            'dimensions': {'width': int(window.width), 'height': int(window.height)},
            'affineTransform': {
                'scaleX': grid.transform.a, 'shearX': 0, 'translateX': x,
                'shearY': 0, 'scaleY': grid.transform.e, 'translateY': y,
            },
            'crsCode': grid.crs,
        },
    }
    failures = throttles = 0
    while True:
        if token is None:
            token = limiter.acquire()
        throttled = False
        try:
            data = _compute_pixels(request)
        except Exception as e:
            kind = _classify_error(e)
            throttled = kind == 'throttle'
            if kind == 'compute_limit':
                raise _ComputeLimitError(str(e)) from e
            if kind == 'fatal':
                raise
            if throttled:
                # throttling is expected while the limiter converges, so it
                # gets a larger retry allowance than genuine failures
                throttles += 1
                if throttles > 4 * max_retries:
                    raise
                # a 429 costs ~1 s and the limiter already gates concurrency, so
                # keep the pause short -- long sleeps just leave a slot idle
                sleep_s = min(4.0, 0.5 * 2 ** min(throttles - 1, 3)) * random.uniform(0.5, 1.5)
            else:
                failures += 1
                if failures > max_retries:
                    raise
                sleep_s = base_backoff * (2 ** (failures - 1))
                print(f"[retry {failures}/{max_retries}] tile {window}: {e} -- backing off {sleep_s:.0f}s")
        else:
            break
        finally:
            limiter.release(token, throttled)
            token = None
        time.sleep(sleep_s)
    # decode after giving the slot back: it's local CPU work, not an EE request
    with rasterio.MemoryFile(data) as mem, mem.open() as ds:
        return ds.read()


def _download_direct(image: ee.Image, grid: _Grid, band_names: list, dtype: str, out_filename: str,
                     max_tile_mb: float, workers: int, max_retries: int, base_backoff: float) -> str:
    """Tiled concurrent download written straight into windows of the output
    GeoTIFF (no temp tiles, no in-memory mosaic). Writes to a temp path and
    renames only once every tile has landed, so a partial file is never left
    at `out_filename`. Raises on the first tile failure."""
    n_bands = len(band_names)
    sched = _GuidedScheduler(n_bands, grid.height, grid.width, dtype, max_tile_mb)
    limiter = _AdaptiveLimiter(workers)
    print(f"Direct download: {grid.width}x{grid.height} px, {n_bands} bands ({dtype}), "
          f"{sched.total_bytes / 2 ** 20:,.1f} MB raw, up to {workers} concurrent requests")

    profile = {
        'driver': 'GTiff', 'width': grid.width, 'height': grid.height, 'count': n_bands,
        'dtype': dtype, 'crs': grid.crs, 'transform': grid.transform, 'nodata': _NODATA[dtype],
        'tiled': True, 'blockxsize': _BLOCK, 'blockysize': _BLOCK, 'interleave': 'band',
        'compress': 'lzw', 'predictor': 3 if dtype.startswith('float') else 2,
        'BIGTIFF': 'IF_SAFER',
    }
    tmp_path = f"{out_filename}.{uuid.uuid4().hex[:8]}.partial"
    write_lock = threading.Lock()
    abort = threading.Event()  # set on the first failure so other workers stop
    errors = []
    progress = {'tiles': 0, 'bytes': 0, 'next_report': 0.1}

    try:
        with rasterio.open(tmp_path, 'w', **profile) as dst:
            def worker():
                while not abort.is_set():
                    token = limiter.acquire()
                    # size the tile only once a slot is free, so it reflects
                    # the concurrency known right now
                    tile = None if abort.is_set() else sched.next(limiter.target)
                    if tile is None:
                        limiter.release(token)
                        return
                    b0, b1, window = tile
                    try:
                        arr = _fetch_tile(image, band_names[b0:b1], grid, window, limiter, token,
                                          max_retries, base_backoff)
                        with write_lock:
                            dst.write(arr, indexes=list(range(b0 + 1, b1 + 1)), window=window)
                            progress['tiles'] += 1
                            progress['bytes'] += arr.nbytes
                            frac = progress['bytes'] / sched.total_bytes
                            if frac >= progress['next_report']:
                                print(f"  {frac:4.0%} ({progress['tiles']} tiles, concurrency {limiter.limit})")
                                progress['next_report'] = math.floor(frac * 10 + 1) / 10
                    except BaseException as e:
                        errors.append(e)
                        abort.set()
                        return

            threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                raise errors[0]
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
        if limiter.limit < workers:
            print(f"  EE throttled concurrency: finished at {limiter.limit}/{workers} concurrent requests")
        os.replace(tmp_path, out_filename)
        return out_filename
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _describe(image: ee.Image, roi: ee.Geometry, crs: str) -> dict:
    """Band names, band types and ROI bounds (in `crs`) -- everything needed to
    plan the download, in a single round trip."""
    return ee.Dictionary({
        'bands': image.bandNames(),
        'types': image.bandTypes(),
        'bounds': roi.bounds(1, crs).coordinates(),
    }).getInfo()


def _download_drive(image: ee.Image, roi: ee.Geometry, out_filename: str, scale: float, crs: str) -> Optional[str]:
    filename_no_ext = os.path.splitext(os.path.basename(out_filename))[0]
    # Use the base name of the output directory as the Drive folder name
    out_dir = os.path.dirname(out_filename)
    drive_folder = os.path.basename(out_dir) if out_dir and os.path.basename(out_dir) else 'CDTS_Downloads'

    task = submit_drive_export(image, filename_no_ext, drive_folder, roi.bounds(), scale=scale, crs=crs)
    print(f"[{filename_no_ext}] Task sent to Google Drive (Task ID: {task.id}). Waiting for completion...")
    return wait_and_download_task(task, drive_folder, filename_no_ext, out_filename)


def download_gee_image(image: ee.Image, roi: ee.Geometry, out_filename: str, method: str = 'auto',
                       scale: float = 30, tile_size: Optional[float] = None, sub_tile_workers: int = 16,
                       crs: str = 'EPSG:4326', max_tile_mb: Optional[float] = None, max_direct_mb: float = 4096,
                       max_retries: int = 5, base_backoff: float = 5.0) -> Optional[str]:
    """
    Downloads an image from GEE to a GeoTIFF, picking the fastest route that
    will work.

    method='direct' fetches the image as a grid of tiles with concurrent
    ee.data.computePixels calls (design adapted from geedim's Tiler): one
    pixel grid is fixed for the whole ROI up front and every tile is requested
    on an exact window of it, so tiles abut with no seams and are written
    straight into their window of the output file -- no temp tiles, no
    re-mosaicking. Tiles are sized in bytes (bands x rows x cols x dtype) and
    split in space first; only deep images are also split by band, so e.g. a
    dense toBands() stack with thousands of bands stays under EE's per-request
    limits while a 6-band composite is never computed twice per area.

    method='drive' submits an Export.image.toDrive batch task and pulls the
    result back (slower to start, but no interactive size/compute limits).

    method='auto' (default) uses 'direct' when the raw image is at most
    `max_direct_mb`, and 'drive' otherwise -- or if 'direct' hits an EE
    interactive compute limit (user memory limit / computation timeout),
    which retrying the same request cannot fix.

    Args:
        image (ee.Image): The Earth Engine image to download.
        roi (ee.Geometry): Region of interest; its bounding box is downloaded.
        out_filename (str): The output file path.
        method (str): 'auto', 'direct' or 'drive'.
        scale (float): Resolution in meters (converted to degrees at the
            equator for a geographic `crs`, as EE itself does).
        tile_size: Deprecated and ignored (tiles are now sized in bytes via
            `max_tile_mb`, not in degrees).
        sub_tile_workers (int): Maximum concurrent computePixels requests for
            this one image. Earth Engine enforces a per-account limit on
            concurrent *interactive* requests (distinct from the batch/Export
            queue; ~40 on a standard tier, only 2-3 for a project in
            noncommercial Restricted Mode) -- see
            https://developers.google.com/earth-engine/guides/usage#concurrent_interactive_requests.
            Concurrency starts at 4, grows while requests succeed and halves
            on every HTTP 429, so it settles just under whatever limit the
            account actually has. If a caller also parallelizes ACROSS images
            (e.g. one process per tile), they all share that account-wide
            limit and will throttle each other.
        crs (str): Output CRS (e.g. 'EPSG:4326', 'EPSG:32723').
        max_tile_mb (float, optional): Upper bound on the raw size of one tile
            request (<= 32, the default). Tiles are sized on demand as
            remaining work / current concurrency (see _GuidedScheduler), so
            this only caps them.
        max_direct_mb (float): In 'auto' mode, images larger than this (raw)
            go straight to Drive export.
        max_retries (int), base_backoff (float): Per-tile retry policy for
            transient failures (network / server errors); throttled requests
            get up to 4 x max_retries short jittered retries.

    Returns:
        str or None: Output filename on success. Returns None (and writes
        nothing to out_filename) if the download failed -- a partial image is
        never written, since a caller relying on file-existence for
        resumability must never see a gappy composite reported as complete.
    """
    if method not in ('auto', 'direct', 'drive'):
        raise ValueError(f"Unknown download method: {method}. Choose 'auto', 'direct' or 'drive'.")
    if tile_size is not None:
        warnings.warn("download_gee_image(tile_size=...) is deprecated and ignored; tiles are sized "
                      "in bytes via max_tile_mb.", DeprecationWarning, stacklevel=2)
    if max_tile_mb is not None and not 0 < max_tile_mb <= _EE_MAX_REQUEST_MB:
        raise ValueError(f"max_tile_mb must be in (0, {_EE_MAX_REQUEST_MB}].")

    if method == 'drive':
        return _download_drive(image, roi, out_filename, scale, crs)

    info = _describe(image, roi, crs)
    band_names = info['bands']
    if not band_names:
        print(f"Image for {out_filename} has no bands -- nothing to download.")
        return None
    dtype = _image_dtype([info['types'][b] for b in band_names])
    ring = info['bounds'][0]
    xs, ys = [c[0] for c in ring], [c[1] for c in ring]
    geographic = rasterio.crs.CRS.from_user_input(crs).is_geographic
    grid = _pixel_grid((min(xs), min(ys), max(xs), max(ys)), scale, crs, geographic)
    size_mb = grid.width * grid.height * len(band_names) * np.dtype(dtype).itemsize / 2 ** 20

    if method == 'auto' and size_mb > max_direct_mb:
        print(f"{out_filename}: {size_mb:,.0f} MB raw exceeds max_direct_mb={max_direct_mb:,.0f} -- using Drive export.")
        return _download_drive(image, roi, out_filename, scale, crs)

    if max_tile_mb is None:
        max_tile_mb = _EE_MAX_REQUEST_MB
    # Cast explicitly so every band slice comes back in the same dtype.
    typed = _CAST[dtype](image)
    t0 = time.time()
    try:
        result = _download_direct(typed, grid, band_names, dtype, out_filename,
                                  max_tile_mb, sub_tile_workers, max_retries, base_backoff)
        print(f"Downloaded {out_filename} ({size_mb:,.1f} MB raw) in {time.time() - t0:.1f}s")
        return result
    except _ComputeLimitError as e:
        if method == 'auto':
            print(f"{out_filename}: direct download hit an EE compute limit ({e}) -- falling back to Drive export.")
            return _download_drive(image, roi, out_filename, scale, crs)
        print(f"Direct download of {out_filename} failed: {e}")
        return None
    except Exception as e:
        print(f"Direct download of {out_filename} failed: {e}")
        return None
