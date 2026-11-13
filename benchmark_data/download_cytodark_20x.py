"""Extract CytoDArk0's 20× original images and labels via ZIP byte ranges.

The public 2.7 GB archive contains many derived crops and maps. This downloads
only the 69 original image/mask pairs plus the authors' split and metadata.
Each ZIP member is CRC-checked by zipfile before it is kept locally.
"""

from __future__ import annotations

import argparse
import io
import shutil
import time
import zipfile
from collections import OrderedDict
from pathlib import Path

import requests

URL = "https://zenodo.org/records/13694738/files/cytoDArk0.zip?download=1"
PREFIX = "cytoDArk0/20x/1024x1024/"
EXTRAS = ("cytoDArk0/20x/1024x1024/folds.csv",
          "cytoDArk0/cytoDArk0_metadata.xlsx")


class HTTPRangeReader(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        for attempt in range(6):
            result = self.session.get(url, headers={"Range": "bytes=-1"}, timeout=90)
            if result.status_code != 429:
                break
            retry_after = result.headers.get("Retry-After", "")
            delay = int(retry_after) if retry_after.isdigit() else min(120, 15 * 2**attempt)
            print(f"Dataset host is rate-limiting; waiting {delay}s…", flush=True)
            time.sleep(delay)
        result.raise_for_status()
        if result.status_code != 206 or "Content-Range" not in result.headers:
            raise RuntimeError("Dataset host does not support ZIP byte ranges")
        self.length = int(result.headers["Content-Range"].rsplit("/", 1)[1])
        self.position = 0
        self.block_size = 16 * 1024 * 1024
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        self.position = (offset if whence == 0 else
                         self.position + offset if whence == 1 else
                         self.length + offset)
        return self.position

    def read(self, count: int = -1) -> bytes:
        if count < 0:
            count = self.length - self.position
        if count == 0 or self.position >= self.length:
            return b""
        chunks = []
        remaining = min(count, self.length - self.position)
        while remaining:
            start = (self.position // self.block_size) * self.block_size
            block = self._block(start)
            offset = self.position - start
            piece = block[offset:offset + remaining]
            if not piece:
                raise RuntimeError("Empty ZIP range block")
            chunks.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(chunks)

    def _block(self, start: int) -> bytes:
        if start in self.cache:
            self.cache.move_to_end(start)
            return self.cache[start]
        end = min(self.length - 1, start + self.block_size - 1)
        for attempt in range(6):
            try:
                result = self.session.get(
                    self.url, headers={"Range": f"bytes={start}-{end}"},
                    timeout=120)
                if result.status_code == 429:
                    retry_after = result.headers.get("Retry-After", "")
                    delay = int(retry_after) if retry_after.isdigit() else min(120, 15 * 2**attempt)
                    print(f"Dataset host is rate-limiting; waiting {delay}s…", flush=True)
                    time.sleep(delay)
                    continue
                result.raise_for_status()
                if result.status_code != 206 or len(result.content) != end - start + 1:
                    raise RuntimeError("Incomplete ZIP byte-range response")
                self.cache[start] = result.content
                if len(self.cache) > 4:
                    self.cache.popitem(last=False)
                return self.cache[start]
            except (requests.RequestException, RuntimeError):
                if attempt == 5:
                    raise
                time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError("Dataset host remained rate-limited")

    def close(self) -> None:
        self.session.close()
        super().close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with HTTPRangeReader(URL) as remote, zipfile.ZipFile(remote) as archive:
        image_names = sorted(n for n in archive.namelist()
                             if n.startswith(PREFIX + "image/") and n.endswith(".png"))
        label_names = sorted(n for n in archive.namelist()
                             if n.startswith(PREFIX + "label/") and n.endswith(".tiff"))
        if len(image_names) != 69 or len(label_names) != 69:
            raise RuntimeError(f"Expected 69 image/mask pairs, got "
                               f"{len(image_names)} images and {len(label_names)} masks")
        for index, name in enumerate(image_names + label_names + list(EXTRAS), 1):
            category = "image" if name in image_names else "label" if name in label_names else "metadata"
            target = root / category / Path(name).name
            if target.is_file():
                continue
            target.parent.mkdir(exist_ok=True)
            partial = target.with_name(target.name + ".part")
            try:
                with archive.open(name) as source, partial.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
                partial.replace(target)
            finally:
                partial.unlink(missing_ok=True)
            if index % 10 == 0 or index == 140:
                print(f"Extracted {index}/140: {target.name}", flush=True)
    print(f"Ready: 69 image/mask pairs in {root}", flush=True)


if __name__ == "__main__":
    main()
