import itertools
import os
import threading

import ee
import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import Affine
from rasterio.windows import Window

from cdts.gee import downloader as dl


# --- pure planning helpers -------------------------------------------------

def test_pixel_grid_geographic_snaps_and_covers_bounds():
    bounds = (-43.6, -23.1, -43.1, -22.6)
    g = dl._pixel_grid(bounds, 30, 'EPSG:4326', geographic=True)
    res = 30 / dl._METERS_PER_DEGREE
    assert g.transform.a == pytest.approx(res) and g.transform.e == pytest.approx(-res)
    # origin on a global multiple of the resolution
    assert g.transform.c / res == pytest.approx(round(g.transform.c / res))
    assert g.transform.f / res == pytest.approx(round(g.transform.f / res))
    # grid fully covers the bounds, by less than one extra pixel per side
    assert g.transform.c <= bounds[0] < g.transform.c + res
    assert g.transform.f >= bounds[3] > g.transform.f - res
    right = g.transform.c + g.width * res
    bottom = g.transform.f - g.height * res
    assert bounds[2] <= right < bounds[2] + 2 * res
    assert bounds[1] >= bottom > bounds[1] - 2 * res


def test_pixel_grid_adjacent_rois_share_grid():
    a = dl._pixel_grid((-43.6, -23.1, -43.35, -22.6), 30, 'EPSG:4326', True)
    b = dl._pixel_grid((-43.35, -23.1, -43.1, -22.6), 30, 'EPSG:4326', True)
    offset_px = (b.transform.c - a.transform.c) / a.transform.a
    assert offset_px == pytest.approx(round(offset_px), abs=1e-6)


def test_pixel_grid_projected_uses_meters():
    g = dl._pixel_grid((600000, 7399980, 630000, 7429980), 30, 'EPSG:32723', geographic=False)
    assert (g.width, g.height) == (1000, 1000)
    assert g.transform == Affine(30, 0, 600000, 0, -30, 7429980)


@pytest.mark.parametrize('types, expected', [
    ([{'precision': 'float'}] * 6, 'float32'),
    ([{'precision': 'double'}, {'precision': 'float'}], 'float64'),
    ([{'precision': 'int', 'min': 0, 'max': 255}], 'uint8'),
    ([{'precision': 'int', 'min': 0, 'max': 65535}, {'precision': 'int', 'min': 0, 'max': 1}], 'uint16'),
    ([{'precision': 'int', 'min': -32768, 'max': 32767}], 'int16'),
    ([{'precision': 'int', 'min': 0, 'max': 65535}, {'precision': 'float'}], 'float32'),
    ([{'precision': 'int', 'min': 0, 'max': 2 ** 32 - 1}, {'precision': 'float'}], 'float64'),
    ([{'precision': 'int', 'min': -2 ** 40, 'max': 2 ** 40}], 'float64'),
])
def test_image_dtype(types, expected):
    assert dl._image_dtype(types) == expected


@pytest.mark.parametrize('bands, h, w, dtype, mb', [
    (6, 7400, 8100, 'float32', 4),       # annual medoid, full WRS scene
    (1, 7400, 8100, 'float32', 4),       # single index band
    (3000, 1856, 1856, 'float32', 4),    # dense toBands() stack
    (3000, 1856, 1856, 'float32', 32),
    (2, 300, 200, 'uint8', 4),           # tiny image -> one tile
    (1, 20000, 20000, 'uint8', 32),      # hits the 10000 px dim cap
    (2000, 100, 100, 'float64', 32),     # hits the 1024 band cap
])
def test_tile_shape_respects_limits(bands, h, w, dtype, mb):
    b, th, tw = dl._tile_shape(bands, h, w, dtype, mb)
    assert 1 <= b <= min(bands, dl._EE_MAX_TILE_BANDS)
    assert 1 <= th <= min(h, dl._EE_MAX_TILE_DIM)
    assert 1 <= tw <= min(w, dl._EE_MAX_TILE_DIM)
    assert b * th * tw * np.dtype(dtype).itemsize <= mb * 2 ** 20
    # spatial dims stay on the tile step unless they span the whole image
    assert th == h or th % dl._TILE_STEP == 0
    assert tw == w or tw % dl._TILE_STEP == 0


def test_tile_shape_splits_space_before_bands():
    # a 6-band composite must not be split into band slices (each slice would
    # re-run the whole composite computation over the same area)
    b, th, tw = dl._tile_shape(6, 7400, 8100, 'float32', 4)
    assert b == 6
    # a deep stack still gets band slices, over a window no smaller than one step
    b, th, tw = dl._tile_shape(3000, 1856, 1856, 'float32', 4)
    assert b < 3000 and th == tw == dl._TILE_STEP


def test_tile_shape_small_image_single_tile():
    assert dl._tile_shape(2, 300, 200, 'uint8', 4) == (2, 300, 200)


def test_tile_shape_tiny_budget_still_fits():
    b, th, tw = dl._tile_shape(3, 1000, 1000, 'float64', 0.01)
    assert b * th * tw * 8 <= 0.01 * 2 ** 20


def _drain(sched, concurrency):
    tiles = []
    while (t := sched.next(concurrency)) is not None:
        tiles.append(t)
    return tiles


@pytest.mark.parametrize('concurrency', [1, 2, 16, 40])
@pytest.mark.parametrize('shape, dtype, mb', [
    ((5, 1300, 1100), 'float32', 32),
    ((6, 1857, 1856), 'float32', 32),
    ((300, 700, 900), 'float32', 4),     # deep stack -> band lanes
    ((1, 12000, 11000), 'uint8', 32),    # wider than the 10000 px dim cap -> column lanes
])
def test_guided_scheduler_covers_exactly_once(shape, dtype, mb, concurrency):
    n_bands, h, w = shape
    sched = dl._GuidedScheduler(n_bands, h, w, dtype, mb)
    cover = np.zeros(shape, dtype=np.uint8)
    for b0, b1, win in _drain(sched, concurrency):
        r, c = int(win.row_off), int(win.col_off)
        assert (b1 - b0) * win.height * win.width * np.dtype(dtype).itemsize <= mb * 2 ** 20
        assert win.width <= dl._EE_MAX_TILE_DIM and win.height <= dl._EE_MAX_TILE_DIM
        cover[b0:b1, r:r + int(win.height), c:c + int(win.width)] += 1
    assert (cover == 1).all()
    assert sched.remaining_bytes == 0


def test_guided_scheduler_follows_concurrency():
    # 6-band composite, 79 MB raw: with 2 concurrent requests allowed the
    # first tiles span whole lanes; with 40 they are small so all run at once
    few = _drain(dl._GuidedScheduler(6, 1857, 1856, 'float32', 32), 2)
    many = _drain(dl._GuidedScheduler(6, 1857, 1856, 'float32', 32), 40)
    assert len(few) <= 8 and len(few) < len(many)
    assert len(many) >= 8
    assert all(b1 - b0 == 6 for b0, b1, _ in few + many)  # never split a composite by band


def test_guided_scheduler_tiles_shrink_towards_the_end():
    sched = dl._GuidedScheduler(1, 10000, 2000, 'float32', 32)
    heights = [int(w.height) for _, _, w in _drain(sched, 4)]
    assert heights[0] > dl._TILE_STEP
    # non-increasing, apart from the last tile of each lane absorbing a sliver
    body = heights[:-4]
    assert body == sorted(body, reverse=True)
    assert min(heights) * 2000 * 4 >= dl._MIN_TILE_MB * 2 ** 20


def test_guided_scheduler_small_image_uses_few_requests():
    # 13 MB single-band image: never cut below _MIN_TILE_MB even when
    # planning for 16 concurrent requests
    tiles = _drain(dl._GuidedScheduler(1, 1857, 1856, 'float32', 32), 16)
    assert len(tiles) <= 4


def test_classify_error():
    assert dl._classify_error(ee.EEException('User memory limit exceeded.')) == 'compute_limit'
    assert dl._classify_error(ee.EEException('Computation timed out.')) == 'compute_limit'
    assert dl._classify_error(ee.EEException('Too many concurrent aggregations.')) == 'throttle'
    assert dl._classify_error(ee.EEException(
        'Too Many Requests: Exceeded Earth Engine concurrency limit. Your project is in Restricted Mode.')) == 'throttle'
    assert dl._classify_error(ee.EEException('Internal error.')) == 'transient'
    assert dl._classify_error(ConnectionError('reset')) == 'transient'
    assert dl._classify_error(ee.EEException("Image.select: Band 'X' not found.")) == 'fatal'


def test_adaptive_limiter_slow_start():
    lim = dl._AdaptiveLimiter(16)
    assert lim.limit == 4
    assert lim.target == 16  # plan for the ceiling until EE pushes back
    for _ in range(4):
        lim.release(lim.acquire())
    assert lim.limit == 8  # doubles per round until the first 429
    for _ in range(100):
        lim.release(lim.acquire())
    assert lim.limit == 16
    assert dl._AdaptiveLimiter(2).limit == 2


def test_adaptive_limiter_backs_off_and_recovers():
    lim = dl._AdaptiveLimiter(16, initial=16)
    # a burst of 5 in-flight requests all throttled counts as one signal
    tokens = [lim.acquire() for _ in range(5)]
    for t in tokens:
        lim.release(t, throttled=True)
    assert lim.limit == 8  # large limits halve
    # a request started under the new limit that is throttled cuts again
    lim.release(lim.acquire(), throttled=True)
    assert lim.limit == 7  # small limits step down by one
    # after a 429, growth is additive: +1 per `limit` successes
    for _ in range(7):
        lim.release(lim.acquire())
    assert lim.limit == 8
    assert lim.target == 7  # tiles are planned for the post-cut level, not the probe
    for _ in range(500):
        lim.release(lim.acquire())
    assert lim.limit == 16
    for _ in range(30):
        lim.release(lim.acquire(), throttled=True)
    assert lim.limit == 1


def test_adaptive_limiter_caps_concurrency():
    lim = dl._AdaptiveLimiter(3, initial=3)
    active, peak, lock = [0], [0], threading.Lock()

    def work():
        token = lim.acquire()
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        threading.Event().wait(0.01)
        with lock:
            active[0] -= 1
        lim.release(token)

    threads = [threading.Thread(target=work) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 3


# --- download path with a fake Earth Engine ---------------------------------

RES = 30 / dl._METERS_PER_DEGREE


class FakeEE:
    """Serves windows of a synthetic (bands, H, W) array as GeoTIFF bytes,
    reading them off the request's affine grid like computePixels does."""

    def __init__(self, data, grid, fail_on=None, fail_times=0, error=None):
        self.data, self.grid = data, grid
        self.names = [f'b{i}' for i in range(data.shape[0])]
        self.fail_on, self.fail_times, self.error = fail_on, fail_times, error
        self.calls = 0
        self.lock = threading.Lock()

    def compute_pixels(self, req):
        with self.lock:
            self.calls += 1
            if self.fail_on is not None and self.fail_times > 0 and self.fail_on(req):
                self.fail_times -= 1
                raise self.error
        g = req['grid']
        at = g['affineTransform']
        col = round((at['translateX'] - self.grid.transform.c) / self.grid.transform.a)
        row = round((at['translateY'] - self.grid.transform.f) / self.grid.transform.e)
        w, h = g['dimensions']['width'], g['dimensions']['height']
        idx = [self.names.index(b) for b in req['bandIds']]
        arr = self.data[idx, row:row + h, col:col + w]
        with MemoryFile() as mem:
            with mem.open(driver='GTiff', width=w, height=h, count=len(idx), dtype=arr.dtype,
                          crs=g['crsCode'], transform=Affine(at['scaleX'], 0, at['translateX'],
                                                             0, at['scaleY'], at['translateY'])) as ds:
                ds.write(arr)
            return mem.read()


@pytest.fixture
def fake_env(monkeypatch):
    def make(shape=(4, 700, 900), dtype='float32', **kw):
        rng = np.random.default_rng(0)
        data = rng.random(shape).astype(dtype)
        x0, y0 = round(-43.5 / RES) * RES, round(-22.7 / RES) * RES  # on the global grid
        grid = dl._Grid('EPSG:4326', Affine(RES, 0, x0, 0, -RES, y0), shape[2], shape[1])
        fake = FakeEE(data, grid, **kw)
        monkeypatch.setattr(dl, '_compute_pixels', fake.compute_pixels)
        monkeypatch.setattr(dl.time, 'sleep', lambda s: None)
        monkeypatch.setitem(dl._CAST, dtype, lambda im: im)
        bounds = (grid.transform.c, grid.transform.f - grid.height * RES,
                  grid.transform.c + grid.width * RES, grid.transform.f)
        info = {'bands': fake.names, 'types': {n: {'precision': 'float'} for n in fake.names},
                'bounds': [[[bounds[0], bounds[1]], [bounds[2], bounds[1]], [bounds[2], bounds[3]],
                            [bounds[0], bounds[3]], [bounds[0], bounds[1]]]]}
        monkeypatch.setattr(dl, '_describe', lambda image, roi, crs: info)
        return fake, data
    return make


def _no_partials(tmp_path):
    return not [f for f in os.listdir(tmp_path) if f.endswith('.partial')]


def test_direct_download_reassembles_exactly(fake_env, tmp_path):
    fake, data = fake_env()
    out = str(tmp_path / 'out.tif')
    # tiny tile budget -> many band + spatial tiles
    assert dl.download_gee_image(object(), object(), out, method='direct', max_tile_mb=0.5,
                                 sub_tile_workers=8) == out
    assert fake.calls > 4
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(), data)
        assert ds.nodata == float('-inf')
        assert ds.descriptions == tuple(fake.names)
        assert ds.transform.a == pytest.approx(RES)
        assert ds.profile['tiled']
    assert _no_partials(tmp_path)


def test_transient_errors_are_retried(fake_env, tmp_path):
    fake, data = fake_env(fail_on=lambda r: True, fail_times=3,
                          error=ee.EEException('Too many concurrent aggregations.'))
    out = str(tmp_path / 'out.tif')
    assert dl.download_gee_image(object(), object(), out, method='direct', max_tile_mb=1) == out
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(), data)


def test_adapts_to_ee_concurrency_limit(fake_env, tmp_path, monkeypatch):
    fake, data = fake_env()
    ee_limit, in_flight, lock = 2, [0], threading.Lock()
    serve = fake.compute_pixels

    def limited(req):
        with lock:
            in_flight[0] += 1
            over = in_flight[0] > ee_limit
        try:
            if over:
                raise ee.EEException('Too Many Requests: Exceeded Earth Engine concurrency limit.')
            threading.Event().wait(0.005)
            return serve(req)
        finally:
            with lock:
                in_flight[0] -= 1

    monkeypatch.setattr(dl, '_compute_pixels', limited)
    out = str(tmp_path / 'out.tif')
    assert dl.download_gee_image(object(), object(), out, method='direct', max_tile_mb=0.25,
                                 sub_tile_workers=16) == out
    with rasterio.open(out) as ds:
        np.testing.assert_array_equal(ds.read(), data)


def test_failed_tile_leaves_no_output(fake_env, tmp_path):
    # first two tiles succeed, the rest fail permanently
    fake_env(fail_on=lambda r, n=itertools.count(): next(n) >= 2, fail_times=99,
             error=ee.EEException("Image.select: Band 'b3' not found."))
    out = str(tmp_path / 'out.tif')
    assert dl.download_gee_image(object(), object(), out, method='direct', max_tile_mb=0.5) is None
    assert not os.path.exists(out)
    assert _no_partials(tmp_path)


def test_existing_output_untouched_on_failure(fake_env, tmp_path):
    fake_env(fail_on=lambda r: True, fail_times=99, error=ee.EEException('bad request'))
    out = tmp_path / 'out.tif'
    out.write_bytes(b'previous')
    assert dl.download_gee_image(object(), object(), str(out), method='direct') is None
    assert out.read_bytes() == b'previous'


def test_auto_falls_back_to_drive_on_compute_limit(fake_env, tmp_path, monkeypatch):
    fake, _ = fake_env(fail_on=lambda r: True, fail_times=99,
                       error=ee.EEException('User memory limit exceeded.'))
    drive_calls = []
    monkeypatch.setattr(dl, '_download_drive', lambda *a: drive_calls.append(a) or a[2])
    out = str(tmp_path / 'out.tif')
    assert dl.download_gee_image(object(), object(), out, method='auto', max_tile_mb=0.5) == out
    assert len(drive_calls) == 1
    # compute-limit errors are not retried tile by tile
    assert fake.calls < 32
    assert _no_partials(tmp_path)


def test_direct_does_not_fall_back(fake_env, tmp_path, monkeypatch):
    fake_env(fail_on=lambda r: True, fail_times=99, error=ee.EEException('User memory limit exceeded.'))
    monkeypatch.setattr(dl, '_download_drive', lambda *a: pytest.fail('drive used'))
    assert dl.download_gee_image(object(), object(), str(tmp_path / 'o.tif'), method='direct') is None


def test_auto_routes_huge_images_to_drive(fake_env, tmp_path, monkeypatch):
    fake, _ = fake_env()
    monkeypatch.setattr(dl, '_download_drive', lambda *a: 'drive')
    assert dl.download_gee_image(object(), object(), str(tmp_path / 'o.tif'), max_direct_mb=1) == 'drive'
    assert fake.calls == 0


def test_invalid_args():
    with pytest.raises(ValueError):
        dl.download_gee_image(object(), object(), 'x.tif', method='ftp')
    with pytest.raises(ValueError):
        dl.download_gee_image(object(), object(), 'x.tif', max_tile_mb=64)
    with pytest.raises(ValueError):
        dl.download_gee_image(object(), object(), 'x.tif', max_tile_mb=0)


def test_tile_size_deprecated(fake_env, tmp_path):
    fake_env()
    with pytest.warns(DeprecationWarning):
        dl.download_gee_image(object(), object(), str(tmp_path / 'o.tif'), method='direct', tile_size=0.25)
