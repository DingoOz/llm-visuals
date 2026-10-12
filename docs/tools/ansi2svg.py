#!/usr/bin/env python3
"""Convert a `tmux capture-pane -e -p` dump (with SGR colour codes) into a
standalone SVG. Used to produce the screenshots in docs/.

    tmux new-session -d -s shot -x 140 -y 44 "llm-visuals"
    sleep 10; tmux capture-pane -t shot -e -p > shot.ansi; tmux kill-session -t shot
    python3 docs/tools/ansi2svg.py shot.ansi docs/dashboard.svg
"""
import html
import re
import sys

CW, CH = 8.4, 17.0  # cell size in px for a 14px monospace font
PAD = 14
BG = "#0d0f16"
ANSI16 = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
          "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#ffffff"]
SGR = re.compile(r"\x1b\[([0-9;]*)m")
# glyph -> (where it starts down the cell, how much of the cell it fills)
BLOCKS = {"\u2580": (0.0, 0.5), "\u2584": (0.5, 0.5), "\u2588": (0.0, 1.0)}


def idx256(n):
    if n < 16:
        return ANSI16[n]
    if n < 232:
        n -= 16
        r, g, b = n // 36, (n // 6) % 6, n % 6
        lv = lambda v: 0 if v == 0 else 55 + v * 40  # noqa: E731
        return "#%02x%02x%02x" % (lv(r), lv(g), lv(b))
    v = 8 + (n - 232) * 10
    return "#%02x%02x%02x" % (v, v, v)


def parse(text):
    rows = []
    # tmux does not repeat a colour that is already in force, and that holds
    # from the end of one line to the start of the next.
    fg, bg, bold = None, None, False
    for line in text.split("\n"):
        cells = []
        pos = 0
        for m in SGR.finditer(line):
            for ch in line[pos:m.start()]:
                cells.append((ch, fg, bg, bold))
            pos = m.end()
            codes = [int(c) if c else 0 for c in m.group(1).split(";")] if m.group(1) else [0]
            i = 0
            while i < len(codes):
                c = codes[i]
                if c == 0:
                    fg, bg, bold = None, None, False
                elif c == 1:
                    bold = True
                elif c == 22:
                    bold = False
                elif 30 <= c <= 37:
                    fg = ANSI16[c - 30]
                elif 90 <= c <= 97:
                    fg = ANSI16[c - 90 + 8]
                elif 40 <= c <= 47:
                    bg = ANSI16[c - 40]
                elif 100 <= c <= 107:
                    bg = ANSI16[c - 100 + 8]
                elif c == 39:
                    fg = None
                elif c == 49:
                    bg = None
                elif c in (38, 48) and i + 1 < len(codes):
                    col = None
                    if codes[i + 1] == 2 and i + 4 < len(codes):
                        col = "#%02x%02x%02x" % tuple(codes[i + 2:i + 5])
                        i += 4
                    elif codes[i + 1] == 5 and i + 2 < len(codes):
                        col = idx256(codes[i + 2])
                        i += 2
                    if c == 38:
                        fg = col
                    else:
                        bg = col
                i += 1
        for ch in line[pos:]:
            cells.append((ch, fg, bg, bold))
        rows.append(cells)
    while rows and not any(c[0].strip() for c in rows[-1]):
        rows.pop()
    return rows


def svg(rows, cols):
    w = PAD * 2 + cols * CW
    h = PAD * 2 + len(rows) * CH
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.0f} {h:.0f}">',
        f'<rect width="100%" height="100%" rx="10" fill="{BG}"/>',
        '<style>text{font-family:"JetBrains Mono","Fira Code","DejaVu Sans Mono",Menlo,monospace;'
        'font-size:14px;white-space:pre}</style>',
    ]
    for y, row in enumerate(rows):  # background runs
        x = 0
        while x < len(row):
            bg = row[x][2]
            x0 = x
            while x < len(row) and row[x][2] == bg:
                x += 1
            if bg and bg != BG:
                out.append(f'<rect x="{PAD + x0 * CW:.1f}" y="{PAD + y * CH:.1f}" '
                           f'width="{(x - x0) * CW + 0.3:.1f}" height="{CH:.1f}" fill="{bg}"/>')
    # Block glyphs are drawn as rectangles: a picture made of half blocks
    # (the expert landscape) shows seams wherever a font's glyph falls short
    # of the cell, and no two fonts agree on where that is.
    for y, row in enumerate(rows):
        x = 0
        while x < len(row):
            ch, fg = row[x][0], row[x][1]
            x0 = x
            while x < len(row) and (row[x][0], row[x][1]) == (ch, fg):
                x += 1
            if ch in BLOCKS:
                top, part = BLOCKS[ch]
                out.append(f'<rect x="{PAD + x0 * CW:.1f}" y="{PAD + (y + top) * CH:.1f}" '
                           f'width="{(x - x0) * CW + 0.3:.1f}" height="{part * CH + 0.3:.1f}" '
                           f'fill="{fg or "#dee2ec"}"/>')
    for y, row in enumerate(rows):  # text runs
        x = 0
        while x < len(row):
            _, fg, _, bold = row[x]
            x0 = x
            run = ""
            while x < len(row) and (row[x][1], row[x][3]) == (fg, bold):
                run += " " if row[x][0] in BLOCKS else row[x][0]
                x += 1
            if run.strip():
                wt = ' font-weight="bold"' if bold else ""
                out.append(f'<text x="{PAD + x0 * CW:.1f}" y="{PAD + y * CH + 13:.1f}" fill="{fg or "#dee2ec"}"{wt} '
                           f'textLength="{len(run) * CW:.1f}" lengthAdjust="spacingAndGlyphs">{html.escape(run)}</text>')
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: ansi2svg.py capture.ansi out.svg")
    rows = parse(open(sys.argv[1], encoding="utf-8", errors="replace").read())
    cols = max(len(r) for r in rows)
    open(sys.argv[2], "w").write(svg(rows, cols))
    print(sys.argv[2], len(rows), "rows", cols, "cols")
