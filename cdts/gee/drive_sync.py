"""
Batch Google Drive export + auto-download for GEE tasks.

Unlike downloader.py's method='direct' (synchronous getDownloadURL calls,
one HTTP round-trip with server-side compute per sub-tile), this submits
ee.batch.Export.image.toDrive tasks that run asynchronously in GEE's own
task queue, then polls task status and pulls the finished file back from
Drive via the Drive API v3 -- reusing whatever OAuth credentials Earth
Engine is already initialized with (which include the 'drive' scope by
default for the standard `earthengine authenticate` flow).
"""

import io
import os
import time
from typing import Optional

import ee


def _drive_service(credentials=None):
    from googleapiclient.discovery import build

    if credentials is None:
        credentials = ee.data.get_persistent_credentials()
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def submit_drive_export(image: "ee.Image", description: str, folder: str, region: "ee.Geometry",
                         scale: float = 30, crs: str = "EPSG:4326") -> "ee.batch.Task":
    """Starts an Export.image.toDrive task and returns it immediately
    (does not block on completion)."""
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=1e13,
    )
    task.start()
    return task


def _find_drive_file(service, folder_name: str, file_prefix: str) -> Optional[dict]:
    folder_q = service.files().list(
        q=f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false",
        fields="files(id, name)",
    ).execute()
    folders = folder_q.get("files", [])
    if not folders:
        return None
    folder_id = folders[0]["id"]

    file_q = service.files().list(
        q=f"'{folder_id}' in parents and name contains '{file_prefix}' and trashed = false",
        fields="files(id, name, size)",
    ).execute()
    files = file_q.get("files", [])
    # Drive's "contains" is a substring match, so e.g. tile "14_064" would
    # also match another tile's "214_064_2001.tif". Narrow to exact (or
    # sharded, "<prefix>-...") matches so overlapping numeric labels can't
    # collide.
    exact = [
        f for f in files
        if f["name"] == f"{file_prefix}.tif" or f["name"].startswith(f"{file_prefix}-")
    ]
    return exact[0] if exact else None


def _download_drive_file(service, file_id: str, out_path: str) -> None:
    from googleapiclient.http import MediaIoBaseDownload

    request = service.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 32)
        done = False
        while not done:
            _status, done = downloader.next_chunk()


def wait_and_download_task(task: "ee.batch.Task", drive_folder: str, file_prefix: str, out_path: str,
                            credentials=None, poll_interval: int = 15, timeout: int = 7200,
                            delete_after: bool = True) -> Optional[str]:
    """Blocks until `task` finishes, then locates the matching file in the
    given Drive folder, downloads it to `out_path`, and (by default)
    deletes the Drive copy so repeated batch runs don't exhaust Drive quota.

    Returns out_path on success, None on failure/timeout.
    """
    service = _drive_service(credentials)
    waited = 0
    while waited < timeout:
        status = task.status()
        state = status.get("state")
        if state == "COMPLETED":
            break
        if state in ("FAILED", "CANCELLED"):
            print(f"[{file_prefix}] Task {state}: {status.get('error_message')}")
            return None
        time.sleep(poll_interval)
        waited += poll_interval
    else:
        print(f"[{file_prefix}] Timed out waiting for task after {timeout}s.")
        return None

    drive_file = _find_drive_file(service, drive_folder, file_prefix)
    if drive_file is None:
        print(f"[{file_prefix}] Task completed but no matching file found in Drive folder '{drive_folder}'.")
        return None

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    _download_drive_file(service, drive_file["id"], out_path)

    if delete_after:
        service.files().delete(fileId=drive_file["id"]).execute()

    return out_path
