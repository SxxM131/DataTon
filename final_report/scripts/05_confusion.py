"""Confusion matrices: OOF (CNN omitted) and test_like C."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

OUT = Path(__file__).resolve().parents[1] / "figures"
CACHE = Path(__file__).resolve().parent / "_cache"
OUT.mkdir(parents=True, exist_ok=True)
labels = [1, 2, 3, 4, 5, 6]

oof = pd.read_csv("/home/ubuntu/soomin/datathon_img/artifacts/oof_C/oof_predictions.csv")
cm_oof = confusion_matrix(oof["label"], oof["pred_post_gate"], labels=labels)
f1_oof = f1_score(oof["label"], oof["pred_post_gate"], average="macro", labels=labels)

tl = pd.read_csv(CACHE / "c_testlike_preds.csv")
cm_tl = confusion_matrix(tl["label"], tl["pred_C"], labels=labels)
f1_tl = f1_score(tl["label"], tl["pred_C"], average="macro", labels=labels)


def plot_cm(ax, cm, title, f1):
    cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    ax.set_title(f"{title}\nMacro F1={f1:.4f}", fontsize=10)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else "black", fontsize=8)
    return im


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
plot_cm(axes[0], cm_oof, "OOF (CNN omitted approx, n=31784)", f1_oof)
im = plot_cm(axes[1], cm_tl, "test_like C (reconstructed, n=6514)", f1_tl)
fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04, label="row-norm")
fig.suptitle(
    "Confusion — sources: oof_C + val_approx/p_cnn reconstruct (matches metrics C F1=0.9349)",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 0.92, 0.92])
fig.savefig(OUT / "fig_confusion.png", dpi=160)
fig.savefig(OUT / "fig_confusion.pdf")
np.save(OUT / "cm_oof.npy", cm_oof)
np.save(OUT / "cm_testlike_C.npy", cm_tl)
print("OOF F1", f1_oof, "test_like F1", f1_tl)
print("wrote", OUT / "fig_confusion.png")
