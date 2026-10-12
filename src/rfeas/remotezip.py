"""Read members of a remote ZIP via HTTP range requests (stdlib only).

zipfile only needs a seekable file object: it reads the end-of-central-directory, then the
central directory, then individual members -- so only those byte ranges are downloaded.
"""
import io
import urllib.request


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str, block: int = 1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
        self.bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _range(self, a: int, b: int) -> bytes:  # inclusive a..b
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.bytes_fetched += len(data)
                return data
            except OSError:
                if attempt == 4:
                    raise

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:  # large member: fetch directly
            data = self._range(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            p = self.pos
            while len(out) < n:
                bi = p // self.block
                if bi not in self.cache:
                    a = bi * self.block
                    self.cache[bi] = self._range(a, min(a + self.block, self.size) - 1)
                    if len(self.cache) > 64:
                        self.cache.pop(next(iter(self.cache)))
                blk = self.cache[bi]
                off = p - bi * self.block
                take = blk[off:off + n - len(out)]
                out += take
                p += len(take)
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)
