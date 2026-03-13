"""Generate minimal valid PNG images for dummy_data smoke testing."""

import os
import struct
import zlib


def make_png(filepath, width=400, height=600):
    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00" + b"\xff" * (width * 3)
    compressed = zlib.compress(raw_data)

    def chunk(ctype, data):
        c = ctype + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    idat = chunk(b"IDAT", compressed)
    iend = chunk(b"IEND", b"")

    with open(filepath, "wb") as f:
        f.write(sig + ihdr + idat + iend)


if __name__ == "__main__":
    for i in range(1, 21):
        make_png(os.path.join("dummy_data", "train", "images", f"r{i:03d}.png"))
    for i in range(1, 11):
        make_png(os.path.join("dummy_data", "test", "images", f"t{i:03d}.png"))
    print("Created 30 dummy PNG images")
