#!/usr/bin/env python3
"""Build a raw RGB565 avatar_set payload for the load_avatar_set MCP tool.

Converts the 14 PNGs in ~/.stackchan/avatar/ (same source layout as
firmware/scripts/avatar_convert/convert_avatars.py) into a single raw
binary matching AvatarSet's layered-mode layout:

    [0 ..)                                      face   x 6
    [kNumFaces * kImageBytes ..)                eyes   x 3
    [(kNumFaces + kNumEyes) * kImageBytes ..)   mouth  x 5

Each frame is 160x120 RGB565 little-endian, no row padding (stride = w*2).
Unlike convert_avatars.py (which emits a compile-time C array for the
firmware build), this emits the raw bytes load_avatar_set expects on the
gateway host filesystem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EMOTIONS = ["idle", "happy", "thinking", "sad", "surprised", "embarrassed"]
EYES = ["eyes_open", "eyes_half", "eyes_closed"]
MOUTHS = ["mouth_closed", "mouth_half", "mouth_open", "mouth_e", "mouth_u"]
DEFAULT_SRC = Path.home() / ".stackchan" / "avatar"
DEFAULT_OUT = Path.home() / ".stackchan" / "avatar_set_layered.raw"
TARGET_W = 160
TARGET_H = 120


def rgb888_to_rgb565_bytes(im) -> bytes:
    if im.mode != "RGB":
        im = im.convert("RGB")
    pixels = im.tobytes()
    out = bytearray(len(pixels) // 3 * 2)
    j = 0
    for i in range(0, len(pixels), 3):
        r, g, b = pixels[i], pixels[i + 1], pixels[i + 2]
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[j] = rgb565 & 0xFF
        out[j + 1] = (rgb565 >> 8) & 0xFF
        j += 2
    return bytes(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--width", type=int, default=TARGET_W)
    ap.add_argument("--height", type=int, default=TARGET_H)
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("Need Pillow: uv run --with pillow python3 build_avatar_set.py")

    total = bytearray()

    def add_one(stem: str) -> None:
        p = args.src / f"{stem}.png"
        if not p.exists():
            sys.exit(f"missing source: {p}")
        im = Image.open(p)
        im = im.resize((args.width, args.height), Image.LANCZOS)
        data = rgb888_to_rgb565_bytes(im)
        expected = args.width * args.height * 2
        assert len(data) == expected, f"{stem}: {len(data)} != {expected}"
        total.extend(data)
        print(f"  {stem:14s} -> {len(data)} bytes")

    print("Faces:")
    for name in EMOTIONS:
        add_one(name)
    print("Eyes:")
    for name in EYES:
        add_one(name)
    print("Mouths:")
    for name in MOUTHS:
        add_one(name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(total)
    print(f"\nTotal: {len(total)} bytes ({len(total)/1024:.1f} KB)")
    print(f"Wrote: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
