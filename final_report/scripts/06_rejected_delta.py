"""Rejected attempts Delta horizontal bar. Sources in facts.md sec.6."""
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
    ("HMM vs softmono (OOF)", 0.8789 - 0.8936),
    ("R1/R2 rules vs A (OOF)", 0.8959 - 0.8872),
    ("pressure B drop meas_p", 0.8864 - 0.8886),
    ("pressure C rel30", 0.8850 - 0.8886),
    ("D = C+transition feats", -0.01002),
    ("E = img10 relative", -0.00740),
    ("redecide23 +pressure", -0.00123),
    ("tau only vs C OOF", -0.00044),
    ("gate56 only vs C OOF", -0.00365),
    ("C ablation (adopted)", +0.00953),
]
names = [r[0] for r in rows]
deltas = [r[1] for r in rows]
colors = ["#54A24B" if d > 0.005 else ("#F58518" if d > 0 else "#E45756") for d in deltas]

fig, ax = plt.subplots(figsize=(9.5, 5.2))
y = list(range(len(names)))[::-1]
ax.barh(list(y), deltas[::-1], color=colors[::-1], height=0.7)
ax.axvline(0, color="#333", lw=0.8)
ax.axvline(0.01, color="#999", ls="--", lw=0.8, label="RULES +0.01")
ax.axvline(0.005, color="#bbb", ls=":", lw=0.8, label="advance +0.005")
ax.set_yticks(list(y))
ax.set_yticklabels(names[::-1])
ax.set_xlabel("Delta Macro F1 vs baseline")
ax.set_title("Rejected attempts (green = adopted C ablation ref)")
ax.legend(loc="lower right", frameon=False, fontsize=8)
for yi, d in zip(y, deltas[::-1]):
    ax.text(d + (0.0008 if d >= 0 else -0.0008), yi, f"{d:+.4f}",
            va="center", ha="left" if d >= 0 else "right", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_rejected_delta.png", dpi=160)
fig.savefig(OUT / "fig_rejected_delta.pdf")
print("wrote", OUT / "fig_rejected_delta.png")
