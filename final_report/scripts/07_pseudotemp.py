"""Original thermal RGB -> pseudo-temperature map (early/late same experiment)."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _font_setup import setup
setup()

CACHE = Path(__file__).resolve().parent / "_cache"
OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
paths = json.loads((CACHE / "img_paths.json").read_text())


def rgb_to_hue_deg(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx = np.max(x, axis=-1)
    mn = np.min(x, axis=-1)
    d = mx - mn
    hue = np.zeros_like(mx)
    mask = d > 1e-8
    m = mask & (mx == r)
    hue[m] = ((g[m] - b[m]) / d[m]) % 6
    m = mask & (mx == g)
    hue[m] = (b[m] - r[m]) / d[m] + 2
    m = mask & (mx == b)
    hue[m] = (r[m] - g[m]) / d[m] + 4
    return (hue * 60.0) % 360.0


def pseudo_temp(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float64) / 255.0
    white = (x.min(axis=-1) > 0.92) & ((x.max(axis=-1) - x.min(axis=-1)) < 0.08)
    hue = rgb_to_hue_deg(rgb)
    pt = np.clip((300.0 - hue) / 300.0, 0.0, 1.0)
    return np.where(white, 1.0, pt)


fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
for col, key, tag in [(0, "early", "early (label1)"), (1, "late", "late (label6)")]:
    im = np.asarray(Image.open(paths[key]).convert("RGB"))
    pt = pseudo_temp(im)
    axes[0, col].imshow(im)
    axes[0, col].set_title(f"RGB — {tag}")
    axes[0, col].axis("off")
    im2 = axes[1, col].imshow(pt, cmap="inferno", vmin=0, vmax=1)
    axes[1, col].set_title(f"pseudo-T pt — mean={pt.mean():.3f}")
    axes[1, col].axis("off")
    fig.colorbar(im2, ax=axes[1, col], fraction=0.046, pad=0.04)

fig.suptitle(
    f"Exp {paths['eid'][:12]}... early/late frames\n"
    "pt=clip((300-hue)/300,0,1), white=1 | STATUS color_stats / validation PNG",
    fontsize=10,
)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT / "fig_pseudotemp.png", dpi=160)
fig.savefig(OUT / "fig_pseudotemp.pdf")
print("wrote", OUT / "fig_pseudotemp.png")
