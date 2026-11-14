"""Model C pipeline block diagram."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, xy, wh, text, fc="#EEF3F8", ec="#2F4B7C"):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                 color="#333333", lw=1.2))


fig, ax = plt.subplots(figsize=(11, 4.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("Model C: DataTon tab + img10 + CNN fuse 0.1 + v7 gates", fontsize=12, pad=8)

box(ax, (0.3, 3.2), (2.2, 1.2), "Sensors / meta / rules\n(past windows in sec)")
box(ax, (0.3, 1.4), (2.2, 1.2), "Thermal PNG\npseudo-T -> img10")
box(ax, (3.0, 3.2), (2.3, 1.2), "LightGBM tab\n(img10 mode)")
box(ax, (3.0, 1.4), (2.3, 1.2), "pairdiff CNN\nResNet18 9ch")
box(ax, (5.8, 2.2), (2.2, 1.4), "Late fusion\nP~Ptab^0.9 * Pcnn^0.1\nargmax")
box(ax, (8.5, 2.0), (2.3, 1.8), "v7 gates\nredecide23\ngate6->gate5\nfloor4->hold")

arrow(ax, (2.5, 3.8), (3.0, 3.8))
arrow(ax, (2.5, 2.0), (3.0, 2.0))
arrow(ax, (5.3, 3.8), (5.8, 3.1))
arrow(ax, (5.3, 2.0), (5.8, 2.7))
arrow(ax, (8.0, 2.9), (8.5, 2.9))

ax.text(5.5, 0.55,
        "Sources: SOOMIN_CHANGES.md / selection.json / DataTon README\n"
        "CSV: img10_fuse01_v7gates.csv  md5 8660d119...",
        ha="center", va="center", fontsize=8, color="#444444")

fig.tight_layout()
fig.savefig(OUT / "fig_pipeline.png", dpi=160)
fig.savefig(OUT / "fig_pipeline.pdf")
print("wrote", OUT / "fig_pipeline.png")
