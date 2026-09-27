"""Generate the Zeit logo (hyperspectral temporal cube with a pixel Z).

Usage:  python docs/scripts/make_logo.py [output_dir]
Default output: docs/assets/logo/

The cube is a stack of N time slices (layers). Each layer is an n x n grid of
pixels coloured as one spectral band (violet -> red, top -> bottom). The Z is
drawn on the front face across the layers: the top layer gives the upper bar,
the middle layers the diagonal and the bottom layer the lower bar.

Every SVG written here is plain, hand-editable SVG: each time slice is its own
<g> (also an Inkscape layer), and the Z pixels carry class="z", so the files
open cleanly in Inkscape, Illustrator, Figma or a text editor.
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "assets", "logo")

# ---- design parameters -------------------------------------------------------
N = 4            # pixels per side == number of time slices
GAP = 0.30       # vertical gap between slices, in pixel units
SPECTRUM = ["#6D3FE0", "#2F6BFF", "#00A9D6", "#1FB36B", "#E9C200", "#FF7A1A", "#E0303A"]
THEMES = {
    "light": dict(ink="#0C3B66", bg="#FFFFFF"),   # for white backgrounds
    "dark": dict(ink="#E6E9EE", bg="#0E151E"),    # for the dark docs theme
}
FACE_SHADE = dict(top=1.0, left=0.88, right=0.70)  # iso shading per face
GRID_STROKE = 1.3

U = 10.0
A = U * math.cos(math.radians(30))


def hx(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def mix(fg, bg, t):
    f, b = hx(fg), hx(bg)
    return "#%02X%02X%02X" % tuple(round(b[k] + (f[k] - b[k]) * t) for k in range(3))


def spectrum(t):
    t = max(0.0, min(1.0, t)) * (len(SPECTRUM) - 1)
    i = min(int(t), len(SPECTRUM) - 2)
    return mix(SPECTRUM[i + 1], SPECTRUM[i], t - i)


def P(x, y, z):
    return ((x - y) * A, (x + y) * U / 2 - z * U)


def pts(p):
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in p)


def z_cells(n):
    path = [(i, 0) for i in range(n)]
    path += [(n - 1 - j, j) for j in range(1, n - 1)]
    path += [(i, n - 1) for i in range(n)]
    return set(path)


def mark(theme):
    """Return (svg_body, bbox) for the cube mark."""
    ink, bg = theme["ink"], theme["bg"]
    Z = z_cells(N)
    out, xs, ys = [], [], []

    def poly(cell, fill, cls=None):
        xs.extend(p[0] for p in cell)
        ys.extend(p[1] for p in cell)
        c = f' class="{cls}"' if cls else ""
        return f'<polygon{c} points="{pts(cell)}" fill="{fill}"/>'

    for j in reversed(range(N)):  # bottom slice first (painter's order)
        band = spectrum(j / (N - 1))
        zb = (N - 1 - j) * (1 + GAP)
        zt = zb + 1
        g = [f'<g id="t{j + 1}" inkscape:groupmode="layer" inkscape:label="time slice {j + 1}">']
        g.append('<g class="top">')
        for yy in range(N):
            for xx in range(N):
                cell = [P(xx, yy, zt), P(xx + 1, yy, zt), P(xx + 1, yy + 1, zt), P(xx, yy + 1, zt)]
                g.append(poly(cell, mix(band, bg, FACE_SHADE["top"])))
        g.append('</g><g class="right">')
        for yy in range(N):
            cell = [P(N, yy, zt), P(N, yy + 1, zt), P(N, yy + 1, zb), P(N, yy, zb)]
            g.append(poly(cell, mix(band, bg, FACE_SHADE["right"])))
        g.append('</g><g class="front">')
        for xx in range(N):
            cell = [P(xx, N, zt), P(xx + 1, N, zt), P(xx + 1, N, zb), P(xx, N, zb)]
            if (xx, j) in Z:
                g.append(poly(cell, ink, "z"))
            else:
                g.append(poly(cell, mix(band, bg, FACE_SHADE["left"])))
        g.append('</g></g>')
        out.append("".join(g))

    body = (f'<g id="mark" stroke="{bg}" stroke-width="{GRID_STROKE}" stroke-linejoin="round">'
            + "".join(out) + "</g>")
    return body, (min(xs), min(ys), max(xs), max(ys))


def wordmark(theme):
    """Monoline lowercase 'zeit'; the i-dot is a single violet pixel. Box: 0..160 x 0..84."""
    ink = theme["ink"]
    st = f'fill="none" stroke="{ink}" stroke-width="10" stroke-linecap="round" stroke-linejoin="round"'
    return ('<g id="wordmark" inkscape:groupmode="layer" inkscape:label="wordmark">'
            f'<path id="z" d="M5 34 H37 L5 74 H37" {st}/>'
            f'<path id="e" d="M50 54 H90 A20 20 0 1 0 84.14 68.14" {st}/>'
            f'<path id="i" d="M110 34 V74" {st}/>'
            f'<rect id="i-dot" x="104" y="9" width="12" height="12" rx="2" fill="{SPECTRUM[0]}"/>'
            f'<path id="t" d="M136 16 V62 A12 12 0 0 0 148 74 H154" {st}/>'
            f'<path id="t-bar" d="M127 36 H152" {st}/>'
            '</g>')


HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" '
        'viewBox="{vb}" width="{w:.0f}" height="{h:.0f}" role="img" aria-label="Zeit">'
        '<title>Zeit</title>')


def doc(vb_x, vb_y, w, h, body):
    return HEAD.format(vb=f"{vb_x:.2f} {vb_y:.2f} {w:.2f} {h:.2f}", w=w * 4, h=h * 4) + body + "</svg>\n"


def build(theme):
    # The cube keeps its original (light) colours on every background; only
    # the wordmark follows the theme.
    body, (x0, y0, x1, y1) = mark(THEMES["light"])
    mw, mh = x1 - x0, y1 - y0
    files = {}
    # 1) mark only, square canvas
    pad = 4
    s = max(mw, mh) + 2 * pad
    files["mark"] = doc(x0 + mw / 2 - s / 2, y0 + mh / 2 - s / 2, s, s, body)
    # 2) horizontal lockup: mark height 104 + wordmark
    k = 104 / mh
    placed = f'<g transform="translate(8,8) scale({k:.4f}) translate({-x0:.2f},{-y0:.2f})">{body}</g>'
    W = mw * k + 26
    files["lockup"] = doc(0, 0, W + 166, 120, placed + f'<g transform="translate({W:.2f},20)">{wordmark(theme)}</g>')
    # 3) stacked lockup: mark above wordmark
    k2 = 150 / mh
    mw2 = mw * k2
    Wt = 240
    placed2 = (f'<g transform="translate({(Wt - mw2) / 2:.2f},10) scale({k2:.4f}) translate({-x0:.2f},{-y0:.2f})">'
               f'{body}</g>')
    files["stacked"] = doc(0, 0, Wt, 260, placed2 + f'<g transform="translate({(Wt - 160) / 2:.2f},172)">{wordmark(theme)}</g>')
    # 4) wordmark only
    files["wordmark"] = doc(-4, 0, 166, 84, wordmark(theme))
    return files


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, theme in THEMES.items():
        for kind, svg in build(theme).items():
            path = os.path.join(OUT, f"zeit-{kind}-{name}.svg")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(svg)
            print("wrote", os.path.relpath(path))
