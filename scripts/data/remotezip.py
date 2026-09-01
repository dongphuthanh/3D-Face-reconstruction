"""Read selected members from a remote zip over HTTP range requests.

Arc2Face ships 35 archives of roughly 28 GB each -- about a terabyte for 21M
images across 1M identities. A fine-tune needs a few tens of thousands of
images, so downloading a whole archive to use 0.5% of it is absurd.

A zip's central directory sits at the END of the file, so with range requests
the index can be read first and then only the wanted members fetched. That turns
a 28 GB download into a few GB.

Implemented as a seekable file-like object rather than a bespoke zip parser, so
the standard library's zipfile handles the format -- including ZIP64, which
these archives require since they exceed 4 GB.
"""

import io
import zipfile

import requests

CHUNK = 8 << 20          # read granularity; the central directory is large
MAX_CACHE = 24


class HttpFile(io.RawIOBase):
    """Minimal seekable reader over an HTTP resource supporting byte ranges."""

    def __init__(self, url, session=None, chunk=CHUNK):
        self.url = url
        self.session = session or requests.Session()
        self.chunk = chunk
        self._pos = 0
        self._cache = {}
        self._order = []

        r = self.session.head(url, allow_redirects=True, timeout=60)
        r.raise_for_status()
        if r.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise IOError(f"{url} does not advertise byte ranges")
        self.size = int(r.headers["Content-Length"])
        # HEAD redirects to a signed CDN URL; reuse it so every read does not
        # repeat the redirect handshake.
        self.url = r.url

    # -- io plumbing ------------------------------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.size + offset
        return self._pos

    def tell(self):
        return self._pos

    def _fetch(self, index):
        if index in self._cache:
            return self._cache[index]
        start = index * self.chunk
        end = min(start + self.chunk, self.size) - 1
        if start > end:
            return b""
        r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                             timeout=120)
        r.raise_for_status()
        data = r.content
        self._cache[index] = data
        self._order.append(index)
        if len(self._order) > MAX_CACHE:
            self._cache.pop(self._order.pop(0), None)
        return data

    def read(self, n=-1):
        if n < 0:
            n = self.size - self._pos
        out = bytearray()
        while n > 0 and self._pos < self.size:
            idx = self._pos // self.chunk
            off = self._pos % self.chunk
            block = self._fetch(idx)
            if not block:
                break
            take = block[off:off + n]
            out.extend(take)
            self._pos += len(take)
            n -= len(take)
        return bytes(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote(url, session=None):
    """zipfile.ZipFile over a remote archive. Reads only the index up front."""
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url, session), buffer_size=CHUNK))
