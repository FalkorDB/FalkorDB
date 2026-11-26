#!/usr/bin/env python3
"""Generate the FalkorDB startup banner from the logo.

Reads build/banner/logo.png (the falkordb.com logo lockup, fk-footer-lockup.svg,
rendered and cropped) and writes the three banner variants src/banner/ embeds:

  truecolor.ans  quadrant blocks, swoosh in 24-bit colour from the logo gradient
  ansi256.ans    the same, swoosh colours snapped to the xterm 256-colour palette
  plain.txt      the same blocks with no colour (NO_COLOR)

There is no ASCII variant: at 80 columns the letters are six rows tall, and
every ASCII mapping tried ran them together, so terminals without the block
glyphs get the one-line text banner src/banner/mod.rs prints instead.

The wordmark is drawn in the terminal's own bold foreground rather than the
logo's navy, so it reads on dark and light themes alike.

Standard library only. Run from the repository root:

    python3 build/banner/generate.py
"""

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "build" / "banner" / "logo.png"
OUT = ROOT / "src" / "banner"

COLS = 80  # banner width in terminal cells
CELL_ASPECT = 2.05  # terminal cell height / width
SUPERSAMPLE = 6  # samples per sub-cell edge
FILL = 0.45  # share of a sub-cell that must be inked to fill it

# Quadrant mask bits: 1 top-left, 2 top-right, 4 bottom-left, 8 bottom-right.
QUADRANTS = " ▘▝▀▖▌▞▛▗▚▐▜▄▙▟█"


def read_png(path):
    """Decode an 8-bit, non-interlaced RGB or RGBA PNG onto white."""
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat = 8, b""
    while pos < len(data):
        (length,) = struct.unpack_from(">I", data, pos)
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        if kind == b"IHDR":
            width, height, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and ctype in (2, 6) and interlace == 0, "unsupported PNG"
        elif kind == b"IDAT":
            idat += body
        pos += 12 + length
    channels = 3 if ctype == 2 else 4
    stride = width * channels
    raw = zlib.decompress(idat)
    rows, prev = [], bytearray(stride)
    for y in range(height):
        kind = raw[y * (stride + 1)]
        line = bytearray(raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)])
        for i in range(stride):
            left = line[i - channels] if i >= channels else 0
            up = prev[i]
            corner = prev[i - channels] if i >= channels else 0
            if kind == 1:
                line[i] = (line[i] + left) & 0xFF
            elif kind == 2:
                line[i] = (line[i] + up) & 0xFF
            elif kind == 3:
                line[i] = (line[i] + (left + up) // 2) & 0xFF
            elif kind == 4:
                p = left + up - corner
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - corner)
                pred = left if pa <= pb and pa <= pc else up if pb <= pc else corner
                line[i] = (line[i] + pred) & 0xFF
        rows.append(line)
        prev = line
    pixels = []
    for line in rows:
        out = []
        for x in range(width):
            px = line[x * channels : x * channels + channels]
            alpha = px[3] / 255 if channels == 4 else 1.0
            out.append(tuple(round(c * alpha + 255 * (1 - alpha)) for c in px[:3]))
        pixels.append(out)
    return width, height, pixels


def is_ink(c):
    return not all(v > 235 for v in c)


def is_wordmark(c):
    # The wordmark is navy rgb(30, 19, 64); the swoosh is saturated.
    return c[0] < 110 and c[1] < 100 and c[2] < 150


def grid(width, height, pixels):
    """Classify every quadrant sub-cell as None, 'w' (wordmark) or an RGB swoosh colour."""
    coords = [(x, y) for y in range(height) for x in range(width) if is_ink(pixels[y][x])]
    x0 = min(x for x, _ in coords)
    x1 = max(x for x, _ in coords)
    y0 = min(y for _, y in coords)
    y1 = max(y for _, y in coords)
    cw = (x1 - x0 + 1) / COLS
    ch = cw * CELL_ASPECT
    rows = int((y1 - y0 + 1) / ch + 0.999)
    cells = []
    for cy in range(rows):
        line = []
        for cx in range(COLS):
            subs = []
            for j in range(2):
                for i in range(2):
                    sx, sy = x0 + cx * cw + i * cw / 2, y0 + cy * ch + j * ch / 2
                    n = dark = 0
                    swoosh = []
                    for a in range(SUPERSAMPLE):
                        for b in range(SUPERSAMPLE):
                            x = int(sx + (a + 0.5) * cw / 2 / SUPERSAMPLE)
                            y = int(sy + (b + 0.5) * ch / 2 / SUPERSAMPLE)
                            n += 1
                            if x >= width or y >= height:
                                continue
                            c = pixels[y][x]
                            if is_wordmark(c):
                                dark += 1
                            elif is_ink(c) and max(c) - min(c) > 60:
                                swoosh.append(c)
                    if (dark + len(swoosh)) / n < FILL:
                        subs.append(None)
                    elif dark >= len(swoosh):
                        subs.append("w")
                    else:
                        subs.append(tuple(sum(c[k] for c in swoosh) // len(swoosh) for k in range(3)))
            line.append(subs)
        cells.append(line)
    return cells


def cell(subs):
    """Reduce four sub-cells to (mask, kind, colour): one glyph, one colour."""
    words = sum(1 for s in subs if s == "w")
    colours = [s for s in subs if s not in (None, "w")]
    if not words and not colours:
        return 0, None, None
    if words >= len(colours):
        mask = sum(1 << i for i, s in enumerate(subs) if s == "w")
        return mask, "w", None
    mask = sum(1 << i for i, s in enumerate(subs) if s not in (None, "w"))
    colour = tuple(sum(c[k] for c in colours) // len(colours) for k in range(3))
    return mask, "s", colour


def xterm256(rgb):
    """Nearest xterm-256 index (6x6x6 cube or grey ramp) by squared distance."""
    levels = [0, 95, 135, 175, 215, 255]
    best = None
    for idx in range(16, 256):
        if idx < 232:
            n = idx - 16
            c = (levels[n // 36], levels[n // 6 % 6], levels[n % 6])
        else:
            v = 8 + 10 * (idx - 232)
            c = (v, v, v)
        d = sum((a - b) ** 2 for a, b in zip(rgb, c))
        if best is None or d < best[0]:
            best = (d, idx)
    return best[1]


def render(cells, glyphs, colour):
    """Lay the cells out as text; `colour(kind, rgb)` gives the SGR to switch to."""
    out = []
    for line in cells:
        text, current = "", None
        for subs in line:
            mask, kind, rgb = cell(subs)
            if mask and colour:
                sgr = colour(kind, rgb)
                if sgr != current:
                    text += "\x1b[0m" + sgr if current else sgr
                    current = sgr
            text += glyphs[mask]
        text = text.rstrip(" ")
        if current:
            text += "\x1b[0m"
        out.append(text)
    return "\n".join(out) + "\n"


def main():
    cells = grid(*read_png(SOURCE))
    OUT.mkdir(parents=True, exist_ok=True)
    bold = "\x1b[1m"
    variants = {
        "truecolor.ans": (QUADRANTS, lambda k, c: bold if k == "w" else "\x1b[38;2;%d;%d;%dm" % c),
        "ansi256.ans": (QUADRANTS, lambda k, c: bold if k == "w" else "\x1b[38;5;%dm" % xterm256(c)),
        "plain.txt": (QUADRANTS, None),
    }
    for name, (glyphs, colour) in variants.items():
        (OUT / name).write_text(render(cells, glyphs, colour), encoding="utf-8")
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main()
