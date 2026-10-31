"""Generate the signtalk software flow graph as SVG + PNG."""
from pathlib import Path
import cairosvg

W, H = 900, 1760
CX = 330                      # main column centre
RAIL = 724                    # return-rail x
BW, BH = 300, 56              # process box
DW, DH = 172, 56              # diamond half-width / half-height

INK = "#1a1a1a"
QUIET = "#6b6b6b"
EDGE = "#8a8a8a"
GRID = "#e4e4e4"
ACCENT = "#2f6fd0"
ACCENT_FILL = "#e8f0fb"
REJECT = "#b4b4b4"
FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"

parts = []
a = parts.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=15, anchor="middle", fill=INK, weight="normal"):
    a(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
      f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}">{esc(s)}</text>')


def pill(cy, label):
    a(f'<rect x="{CX-90}" y="{cy-24}" width="180" height="48" rx="24" '
      f'fill="none" stroke="{EDGE}" stroke-width="1.5"/>')
    text(CX, cy + 5, label, 15, weight="bold")


def box(cy, label, sub=None, hot=False):
    fill = ACCENT_FILL if hot else "#fafafa"
    stroke = ACCENT if hot else EDGE
    sw = 2 if hot else 1.5
    h = BH if sub is None else BH + 18
    a(f'<rect x="{CX-BW//2}" y="{cy-h//2}" width="{BW}" height="{h}" rx="8" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    if sub is None:
        text(CX, cy + 5, label, 15, weight="600" if hot else "normal")
    else:
        text(CX, cy - 4, label, 15, weight="600" if hot else "normal")
        text(CX, cy + 17, sub, 12.5, fill=QUIET)
    return h


def diamond(cy, label, sub=None):
    pts = f"{CX-DW},{cy} {CX},{cy-DH} {CX+DW},{cy} {CX},{cy+DH}"
    a(f'<polygon points="{pts}" fill="none" stroke="{EDGE}" stroke-width="1.5"/>')
    if sub is None:
        text(CX, cy + 5, label, 14.5)
    else:
        text(CX, cy - 4, label, 14.5)
        text(CX, cy + 15, sub, 12, fill=QUIET)


def vline(y1, y2, arrow=True):
    m = ' marker-end="url(#ar)"' if arrow else ''
    a(f'<path d="M{CX} {y1}V{y2}" fill="none" stroke="{EDGE}" '
      f'stroke-width="1.5"{m}/>')


def reject(y, label):
    """Branch right from the main column out to the return rail."""
    a(f'<path d="M{CX+DW} {y}H{RAIL}" fill="none" stroke="{REJECT}" '
      f'stroke-width="1.5" stroke-dasharray="5 4"/>')
    text(CX + DW + 14, y - 10, label, 12, anchor="start", fill=QUIET)


# ---------------------------------------------------------------- header
a(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
text(48, 48, "signtalk — per-frame control flow", 21, anchor="start", weight="bold")
text(48, 72, "Every rejection path returns to capture without emitting a word",
     13.5, anchor="start", fill=QUIET)

# ---------------------------------------------------------------- nodes
Y_START = 128
Y_CAP = 212
Y_MP = 306
Y_ASM = 404
Y_VEL = 502
Y_SEG = 616
Y_PRE = 742
Y_INF = 840
Y_CONF = 954
Y_ADMIT = 1092
Y_REFR = 1230
Y_BUF = 1352
Y_SIL = 1468
Y_FLUSH = 1596
Y_RAIL_BOTTOM = 1690

pill(Y_START, "START")
vline(Y_START + 24, Y_CAP - 28)

box(Y_CAP, "Capture frame", "640×480 @ 15–25 fps")
vline(Y_CAP + 37, Y_MP - 37)

box(Y_MP, "MediaPipe Tasks", "hands + pose (+ face)")
vline(Y_MP + 37, Y_ASM - 37)

box(Y_ASM, "Assemble (543, 3)", "missing landmarks → NaN, never 0")
vline(Y_ASM + 37, Y_VEL - 37)

box(Y_VEL, "Wrist velocity", "normalised by shoulder width")
vline(Y_VEL + 37, Y_SEG - DH)

diamond(Y_SEG, "Segment complete?", "motion gate FSM")
reject(Y_SEG, "no — keep watching")
vline(Y_SEG + DH, Y_PRE - 37)
text(CX + 10, Y_SEG + DH + 20, "yes", 12, anchor="start", fill=QUIET)

box(Y_PRE, "Preprocess window", "centre, scale, resample to T")
vline(Y_PRE + 37, Y_INF - 37)

box(Y_INF, "ONNX inference", "logits → softmax", hot=True)
vline(Y_INF + 37, Y_CONF - DH)

diamond(Y_CONF, "Confidence ≥ floor?", "open-set rejection")
reject(Y_CONF, "no — discard")
vline(Y_CONF + DH, Y_ADMIT - DH)
text(CX + 10, Y_CONF + DH + 20, "yes", 12, anchor="start", fill=QUIET)

diamond(Y_ADMIT, "Stable or band?", "admission policy")
reject(Y_ADMIT, "no — hold")
vline(Y_ADMIT + DH, Y_REFR - DH)
text(CX + 10, Y_ADMIT + DH + 20, "yes", 12, anchor="start", fill=QUIET)

diamond(Y_REFR, "Within refractory?", "same word just spoken")
reject(Y_REFR, "yes — suppress")
vline(Y_REFR + DH, Y_BUF - 28)
text(CX + 10, Y_REFR + DH + 20, "no", 12, anchor="start", fill=QUIET)

box(Y_BUF, "Append gloss to phrase buffer")
vline(Y_BUF + 28, Y_SIL - DH)

diamond(Y_SIL, "Silence > 1.5 s?", "utterance boundary")
reject(Y_SIL, "no — keep buffering")
vline(Y_SIL + DH, Y_FLUSH - 37)
text(CX + 10, Y_SIL + DH + 20, "yes", 12, anchor="start", fill=QUIET)

box(Y_FLUSH, "Flush buffer → play WAVs", "pre-rendered, no runtime TTS",
    hot=True)

# ------------------------------------------------- return rail (one path)
a(f'<path d="M{CX} {Y_FLUSH+37}V{Y_RAIL_BOTTOM}H{RAIL}V{Y_CAP}H{CX+BW//2}" '
  f'fill="none" stroke="{REJECT}" stroke-width="1.5" stroke-dasharray="5 4" '
  f'marker-end="url(#ar-q)"/>')
text(RAIL - 14, Y_RAIL_BOTTOM - 12, "next frame", 12.5, anchor="end",
     fill=QUIET)

# ---------------------------------------------------------------- legend
LGY = 1724
a(f'<line x1="48" y1="{LGY-16}" x2="{W-48}" y2="{LGY-16}" stroke="{GRID}"/>')
a(f'<rect x="48" y="{LGY-4}" width="22" height="11" rx="3" '
  f'fill="{ACCENT_FILL}" stroke="{ACCENT}" stroke-width="1.5"/>')
text(80, LGY + 6, "compute-heavy stage", 12.5, anchor="start", fill=QUIET)
a(f'<line x1="260" y1="{LGY+1}" x2="300" y2="{LGY+1}" stroke="{REJECT}" '
  f'stroke-width="1.5" stroke-dasharray="5 4"/>')
text(310, LGY + 6, "no word emitted — return to capture", 12.5,
     anchor="start", fill=QUIET)

defs = (
    f'<defs>'
    f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0 0L10 5L0 10z" fill="{EDGE}"/></marker>'
    f'<marker id="ar-q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
    f'markerHeight="6" orient="auto-start-reverse">'
    f'<path d="M0 0L10 5L0 10z" fill="{REJECT}"/></marker>'
    f'</defs>'
)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
       f'viewBox="0 0 {W} {H}">{defs}{"".join(parts)}</svg>')

out = Path("/home/claude/signtalk")
(out / "signtalk_flow.svg").write_text(svg)
cairosvg.svg2png(bytestring=svg.encode(), write_to=str(out / "signtalk_flow.png"),
                 output_width=W * 2, output_height=H * 2, background_color="white")
print("wrote signtalk_flow.svg and signtalk_flow.png")
