"""Official acquisition, restartable extraction, and archive path checks."""
from __future__ import annotations

import re
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import PurePosixPath

from .common import ProjectError, inside, log, now, read_json, relative, sha256, write_json

REPOSITORY = "https://github.com/BinaLab/FloodNet-Supervised_v1.0"
README = "https://raw.githubusercontent.com/BinaLab/FloodNet-Supervised_v1.0/main/README.md"


def download() -> None:
    directory = inside("data/downloads")
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "FloodNet-Supervised_v1.0.zip"
    record_path = directory / "source.json"
    if archive.exists() and record_path.exists():
        record = read_json(record_path)
        if archive.stat().st_size == record["archive_bytes"] and sha256(archive) == record["archive_sha256"]:
            log("Verified existing archive; download skipped.")
            extract(archive)
            return
        raise ProjectError("Existing archive does not match source.json; preserve it and investigate before retrying.")
    saved_readme = directory / "official_README.md"
    if saved_readme.exists():
        readme = saved_readme.read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(README, timeout=60) as response:
            readme = response.read().decode("utf-8")
        saved_readme.write_text(readme, encoding="utf-8")
    links = re.findall(r"https://www\.dropbox\.com/[^\s)]+", readme)
    if len(links) != 1:
        raise ProjectError("Cannot identify one official Dropbox dataset link. Inspect the saved official README.")
    source = links[0]
    url = re.sub(r"([?&])dl=\d", r"\g<1>dl=1", source)
    if "dl=" not in url:
        url += ("&" if "?" in url else "?") + "dl=1"
    partial = archive.with_suffix(".zip.part")
    if partial.exists() and zipfile.is_zipfile(partial):
        with zipfile.ZipFile(partial) as zipped:
            safe_members(zipped)
        log("Completed partial archive found; validating and promoting without another download.")
        finish_download(partial, archive, record_path, source, url)
        return
    last_error = None
    for attempt in range(1, 6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "FloodNetCourseProject/1.0", "Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = response.status == 206 and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                if offset and not resumed:
                    log("Server did not honour resume; restarting the partial archive.")
                    offset = 0
                length = int(response.headers.get("Content-Length", "0"))
                total = offset + length if length else None
                if total and shutil.disk_usage(directory).free < total - offset + 2**30:
                    raise ProjectError("Insufficient disk space for the archive and safety margin.")
                log(f"Downloading official archive: offset={offset:,}; expected bytes={total}.")
                last_log = time.monotonic()
                with partial.open("ab" if resumed else "wb") as stream:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        stream.write(block)
                        if time.monotonic() - last_log > 20:
                            log(f"Downloaded {stream.tell() / 2**30:.2f} GiB" + (f" / {total / 2**30:.2f} GiB" if total else ""))
                            last_log = time.monotonic()
                if total and partial.stat().st_size != total:
                    raise OSError("Incomplete download; partial file retained.")
            with zipfile.ZipFile(partial) as zipped:
                safe_members(zipped)
            finish_download(partial, archive, record_path, source, url)
            return
        except ProjectError:
            raise
        except Exception as error:
            last_error = error
            log(f"Download attempt {attempt}/5: {error}; partial file retained.")
            if attempt < 5:
                time.sleep(min(2**attempt, 16))
    raise ProjectError(f"Official download failed: {last_error}")


def finish_download(partial, archive, record_path, source, url):
    partial.replace(archive)
    write_json(record_path, {"repository": REPOSITORY, "source_url": source, "download_url": url,
               "retrieved_at": now(), "licence": "CDLA-Permissive-1.0",
               "licence_url": "https://cdla.io/permissive-1-0/", "archive": relative(archive),
               "archive_bytes": archive.stat().st_size, "archive_sha256": sha256(archive)})
    extract(archive)


def safe_members(zipped: zipfile.ZipFile):
    # Dropbox folder ZIPs use one '/' directory entry for the archive root.
    # Ignore only that exact empty root sentinel; all other absolute paths remain invalid.
    members = [info for info in zipped.infolist() if not (info.filename == "/" and info.is_dir())]
    seen = set()
    for info in members:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        mode = info.external_attr >> 16
        if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts) or stat.S_ISLNK(mode):
            raise ProjectError(f"Unsafe archive member: {info.filename}")
        key = path.as_posix().casefold().rstrip("/")
        if key in seen:
            raise ProjectError(f"Duplicate archive member: {info.filename}")
        seen.add(key)
    return members


def extract(archive) -> None:
    import zlib
    raw = inside("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = safe_members(zipped)
        pending = sum(i.file_size for i in members if not i.is_dir() and not (raw / i.filename).exists())
        if shutil.disk_usage(raw).free < pending + 2**30:
            raise ProjectError("Insufficient disk space for extracted originals.")
        for index, info in enumerate(members):
            target = inside(raw / info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                crc = 0
                with target.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if target.stat().st_size != info.file_size or crc != info.CRC:
                    raise ProjectError(f"Original differs from archive: {target}. Originals will not be overwritten.")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".extracting")
            with zipped.open(info) as source, temporary.open("wb") as dest:
                shutil.copyfileobj(source, dest, 4 * 1024 * 1024)
            temporary.replace(target)
            if index % 250 == 0:
                log(f"Extracted {index + 1}/{len(members)} archive members")
    write_json("data/manifests/extraction.json", {"complete": True, "completed_at": now(), "members": len(members)})
    log("Official archive extracted; originals preserved.")
