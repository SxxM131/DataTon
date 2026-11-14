"""Transition alignment: copy 4->5 and 5->6 from artifacts."""
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

SRC = Path("/home/ubuntu/soomin/datathon_img/artifacts/transition_plots")
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

for name in ["align_4to5.png", "align_5to6.png", "align_2to3.png", "align_3to4.png"]:
    shutil.copy2(SRC / name, OUT / f"src_{name}")

lead = pd.read_csv("/home/ubuntu/soomin/datathon_img/artifacts/transition_lead_scores.csv")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, name, title, b in [
    (axes[0], "align_4to5.png", "4->5 aligned", "4to5"),
    (axes[1], "align_5to6.png", "5->6 aligned", "5to6"),
]:
    ax.imshow(mpimg.imread(SRC / name))
    ax.axis("off")
    sub = lead[lead.boundary == b].sort_values("early_delta", ascending=False).iloc[0]
    ax.set_title(
        f"{title}\nlead {sub.feat} early_d={sub.early_delta:.3f} (n={int(sub.n_exp)})",
        fontsize=10,
    )

fig.suptitle(
    "Image transition alignment (label onset=0s). "
    "Source: artifacts/transition_plots + transition_lead_scores.csv",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(OUT / "fig_transition_align.png", dpi=160)
fig.savefig(OUT / "fig_transition_align.pdf")
print("wrote", OUT / "fig_transition_align.png")
