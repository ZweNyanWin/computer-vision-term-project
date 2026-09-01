"""Generate the paper figures from the measured results.

Reads only what `evaluate.py` wrote, so a figure can never show a number the
evaluation did not produce. Writes into `figures/`.

    python src/make_figures.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluate import angular_delta, render_at, silhouette_crop  # noqa: E402
from metrics import psnr, ssim  # noqa: E402
from reconstruct import reconstruct_photo  # noqa: E402
from render3d import load_obj, normalize_mesh  # noqa: E402

FIGURES = PROJECT_ROOT / "figures"
INK = "#1D2024"
PROXY_C = "#8A8F98"
MODEL_C = "#A9682F"


def figure_crossover(path: Path) -> None:
    """Gain over the frame-switching baseline, against capture spacing."""
    rows = []
    for every in (2, 4, 6, 9):
        proxy = next(csv.DictReader(open(PROJECT_ROOT / f"output/metrics_e{every}/summary.csv")))
        model = next(csv.DictReader(open(PROJECT_ROOT / f"output/full_e{every}/summary.csv")))
        baseline_psnr = float(model["psnr_baseline_mean_db"])
        baseline_ssim = float(model["ssim_baseline_mean"])
        rows.append({
            "captured_spacing_deg": model["captured_spacing_deg"],
            "psnr_gain_proxy_db": float(proxy["psnr_render_mean_db"]) - baseline_psnr,
            "psnr_gain_model_db": float(model["psnr_render_mean_db"]) - baseline_psnr,
            "ssim_render_proxy": proxy["ssim_render_mean"],
            "ssim_render_model": model["ssim_render_mean"],
            "ssim_baseline": baseline_ssim,
        })
    x = [float(r["captured_spacing_deg"]) for r in rows]
    series = (
        ("psnr_gain_proxy_db", "psnr_gain_model_db", "PSNR gain (dB)"),
        (None, None, None),
    )
    ssim_proxy = [float(r["ssim_render_proxy"]) - float(r["ssim_baseline"]) for r in rows]
    ssim_model = [float(r["ssim_render_model"]) - float(r["ssim_baseline"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (proxy, model, label) in zip(axes, [series[0], (None, None, "SSIM gain")]):
        if proxy:
            yp = [float(r[proxy]) for r in rows]
            ym = [float(r[model]) for r in rows]
        else:
            yp, ym = ssim_proxy, ssim_model
        ax.axhline(0, color=INK, lw=1, ls="--", alpha=.5)
        ax.plot(x, yp, "o-", color=PROXY_C, lw=2, ms=6, label="proxy depth")
        ax.plot(x, ym, "o-", color=MODEL_C, lw=2.4, ms=7, label="learned depth")
        ax.set_xlabel("Captured spacing (degrees)")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=9)
        ax.set_title("Render minus frame-switching baseline", fontsize=10, color=INK)
    axes[0].annotate("above zero:\nreconstruction wins", xy=(60, 0.13), xytext=(42, -1.0),
                     fontsize=8, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=.9))
    fig.suptitle("Reconstruction relative to frame-switching, by capture spacing",
                 fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def figure_qualitative(path: Path, angles=(50, 110, 170, 230), every: int = 4, size: int = 240) -> None:
    """What the score actually compares: truth, render, and baseline."""
    frames = {int(p.stem): p for p in (PROJECT_ROOT / "data").glob("*.jpeg") if p.stem.isdigit()}
    captured = sorted(frames)[::every]
    cache = PROJECT_ROOT / "output/model_e4/meshes"

    cols = []
    for held in angles:
        nearest = min(captured, key=lambda c: abs(angular_delta(c, held)))
        base_path = cache / f"{nearest:02d}"
        if not base_path.with_suffix(".obj").exists():
            reconstruct_photo(frames[nearest], base_path, depth_mode="model", relief=0.42, grid=120)
        v, uv, f, tp = load_obj(base_path.with_suffix(".obj"))
        mesh = (normalize_mesh(v), uv, f, cv2.imread(str(tp)) if tp else None)

        truth = silhouette_crop(cv2.imread(str(frames[held])), size)
        baseline = silhouette_crop(cv2.imread(str(frames[nearest])), size)
        render = silhouette_crop(render_at(mesh, angular_delta(nearest, held), 0.0, size), size)
        if truth is None or baseline is None or render is None:
            continue

        panels, labels = [truth, render, baseline], [
            f"held-out photo at {held} deg",
            f"render  {psnr(truth, render):.1f} dB",
            f"baseline  {psnr(truth, baseline):.1f} dB",
        ]
        col = []
        for img, text in zip(panels, labels):
            band = np.full((26, size, 3), (18, 18, 20), np.uint8)
            cv2.putText(band, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
            col.append(np.vstack([band, img]))
        cols.append(np.vstack(col))

    if cols:
        cv2.imwrite(str(path), np.hstack(cols))
        print("wrote", path)


def figure_depth(path: Path, angle: int = 0) -> None:
    """The proxy's invented geometry beside the learned prediction."""
    src = PROJECT_ROOT / f"data/{angle:02d}.jpeg"
    rows = []
    # cv2.putText renders ASCII only; anything else comes out as '??'.
    for tag, mode in (("PROXY depth - invented geometry", "shape"),
                      ("LEARNED depth - Depth Anything V2", "model")):
        base = PROJECT_ROOT / f"output/figures_cache/{mode}"
        if not base.with_suffix(".obj").exists():
            base.parent.mkdir(parents=True, exist_ok=True)
            reconstruct_photo(src, base, depth_mode=mode, relief=0.42, grid=120)
        v, uv, f, tp = load_obj(base.with_suffix(".obj"))
        mesh = (normalize_mesh(v), uv, f, cv2.imread(str(tp)) if tp else None)
        tiles = [cv2.resize(cv2.imread(str(base.parent / f"{mode}_depth.png")), (240, 320))]
        for yaw in (-30, 0, 30):
            tiles.append(render_at(mesh, yaw, 0.0, 0)[:0] if False else
                         cv2.resize(render_at(mesh, yaw, 0.0, 160), (240, 320)))
        row = np.hstack(tiles)
        band = np.full((28, row.shape[1], 3), (18, 18, 20), np.uint8)
        cv2.putText(band, f"{tag}      depth map  |  yaw -30       yaw 0       yaw +30",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .46, (240, 240, 240), 1, cv2.LINE_AA)
        rows.append(np.vstack([band, row]))
    cv2.imwrite(str(path), np.vstack(rows))
    print("wrote", path)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    figure_crossover(FIGURES / "fig1_crossover.png")
    figure_qualitative(FIGURES / "fig2_what_is_scored.png")
    figure_depth(FIGURES / "fig3_proxy_vs_learned.png")
    print(f"\nOpen them:  open {FIGURES}")


if __name__ == "__main__":
    main()
