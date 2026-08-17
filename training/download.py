#!/usr/bin/env python3
"""Download CID (Cow Images Dataset) from S3."""
import os
import subprocess
import sys

from .config import CFG

SOURCES = {
    "images.tar.gz": "https://cid-21.s3.amazonaws.com/images.tar.gz",
    "yt_images.tar.gz": "https://cid-21.s3.amazonaws.com/yt_images.tar.gz",
    "dataset.csv": "https://cid-21.s3.amazonaws.com/dataset.csv",
}


def download():
    raw = CFG["data_raw"]
    for name, url in SOURCES.items():
        path = os.path.join(raw, name)
        if os.path.exists(path):
            print(f"  [skip] {name} already exists")
            continue
        print(f"  Downloading {name}...")
        subprocess.run(["curl", "-L", "--progress-bar", "-o", path, url], check=True)
        print(f"  [ok] {name}")


def extract():
    import tarfile
    raw = CFG["data_raw"]
    for name in ["images.tar.gz", "yt_images.tar.gz"]:
        path = os.path.join(raw, name)
        if not os.path.exists(path):
            print(f"  [skip] {name} not found")
            continue
        with tarfile.open(path, "r:gz") as t:
            t.extractall(raw)
        print(f"  Extracted {name}")


if __name__ == "__main__":
    download()
    extract()
