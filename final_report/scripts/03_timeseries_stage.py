"""Val experiment sensors + true/pred stage bands."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

CACHE = Path(__file__).resolve().parent / "_cache"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CACHE / "timeseries_E_1a6e_with_pred.csv").sort_values("relative_time_s")
eid = df["experiment_id"].iloc[0]
t = df["relative_time_s"].to_numpy()
y = df["label"].to_numpy().astype(int)
yp = df["pred_C"].to_numpy().astype(int)
cmap = ListedColormap(["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"])

fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True,
                         gridspec_kw={"height_ratios": [0.55, 1.6, 1.2, 1.2]})

ax = axes[0]
ax.set_yticks([])
ax.set_ylabel("true", fontsize=9)
for i in range(len(t) - 1):
    ax.axvspan(t[i], t[i + 1], color=cmap(y[i] - 1), alpha=0.9, lw=0)
ax.axvspan(t[-1], t[-1] + 5, color=cmap(y[-1] - 1), alpha=0.9, lw=0)
axp = ax.twinx()
axp.set_yticks([])
axp.set_ylabel("pred C", fontsize=9)
for i in range(len(t) - 1):
    axp.axvspan(t[i], t[i + 1], ymin=0, ymax=0.35, color=cmap(yp[i] - 1), alpha=0.95, lw=0)
axp.set_ylim(0, 1)

axes[1].plot(t, df["meas_t1"], label="meas_t1", color="#E45756", lw=1.2)
axes[1].plot(t, df["meas_t2"], label="meas_t2", color="#F58518", lw=1.0, alpha=0.85)
axes[1].set_ylabel("Temp (C)")
axes[1].legend(loc="upper left", frameon=False, fontsize=8)
axes[2].plot(t, df["meas_v"], color="#4C78A8", lw=1.1)
axes[2].set_ylabel("Voltage (V)")
axes[3].plot(t, df["meas_p"], color="#54A24B", lw=1.1)
axes[3].set_ylabel("Pressure")
axes[3].set_xlabel("relative_time_s")

handles = [mpatches.Patch(color=cmap(i), label=str(i + 1)) for i in range(6)]
axes[0].legend(handles=handles, loc="upper right", ncol=6, fontsize=8, frameon=False, title="stage")

acc = (y == yp).mean()
fig.suptitle(
    f"Val exp {eid[:10]}... | n={len(df)} | C match={acc:.3f}\n"
    "Source: validation.csv + C preds reconstructed from artifacts (no retrain)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT / "fig_timeseries_stage.png", dpi=160)
fig.savefig(OUT / "fig_timeseries_stage.pdf")
print("wrote", OUT / "fig_timeseries_stage.png", "acc", acc)
