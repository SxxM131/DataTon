"""LB score history bar. Sources: NOTES.md / review.md / user-stated ([confirm])."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    ("hgb softmono", 0.8597, True),
    ("ens RM", 0.8474, True),
    ("xgb RM", 0.8245, False),
    ("dataton_final v5", 0.9375, True),
    ("DataTon v7", 0.9497, True),
    ("C img10+fuse0.1", 0.9540, False),
]
labels = [r[0] for r in rows]
scores = [r[1] for r in rows]
colors = ["#4C78A8" if r[2] else "#E45756" for r in rows]

fig, ax = plt.subplots(figsize=(9.5, 4.2))
bars = ax.bar(range(len(scores)), scores, color=colors, width=0.72)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=18, ha="right")
ax.set_ylabel("Public LB Macro F1")
ax.set_ylim(0.80, 0.98)
ax.set_title("Leaderboard score history")
for b, s, conf in zip(bars, scores, [r[2] for r in rows]):
    ax.text(b.get_x() + b.get_width() / 2, s + 0.004,
            f"{s:.4f}" if conf else f"{s:.4f}*", ha="center", va="bottom", fontsize=9)
ax.legend(
    handles=[
        mpl.patches.Patch(color="#4C78A8", label="from local files"),
        mpl.patches.Patch(color="#E45756", label="* needs confirm"),
    ],
    loc="lower right", frameon=False,
)
fig.tight_layout()
fig.savefig(OUT / "fig_lb_history.png", dpi=160)
fig.savefig(OUT / "fig_lb_history.pdf")
print("wrote", OUT / "fig_lb_history.png")
