import os, hashlib
from typing import Tuple
from fastapi import UploadFile

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def sha256_of_file(fp) -> str:
    hasher = hashlib.sha256()
    fp.seek(0)
    for chunk in iter(lambda: fp.read(1024 * 1024), b""):
        hasher.update(chunk)
    fp.seek(0)
    return "sha256:" + hasher.hexdigest()

def save_upload_file(file: UploadFile, dest_path: str) -> Tuple[int, str]:
    """Save UploadFile to dest_path. Returns (size_bytes, absolute_path)."""
    ensure_dir(os.path.dirname(dest_path))
    size = 0
    with open(dest_path, "wb") as out:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            out.write(chunk)
    # reset pointer for checksum or further reads
    file.file.seek(0)
    return size, os.path.abspath(dest_path)