"""Score a trained splat on views it never saw.

`mlx3d-eval` reports PSNR/SSIM on the *training* views. That is fit, not novel-view
accuracy, and `NEURAL_CAPTURE.md` forbids reporting it as the latter. This computes
the held-out number instead: the splat is rendered at the withheld cameras' exact
recovered poses and intrinsics and scored against the withheld photographs.

What this is, named precisely, because the obvious name is wrong: **held-out
photometric evaluation over a shared SfM initialisation**. Two separate facts:

  * The render and the photograph are pixel-aligned by construction - same
    extrinsics, same intrinsics, same image grid - so they are compared full-frame
    with no silhouette cropping or scale normalisation to flatter either one. The
    withheld views are absent from the training model, so no Gaussian is ever
    fitted to one of their pixels.
  * But the initialisation is **not** independent of them. Structure-from-motion
    was solved over all 88 photographs before the split was applied, so on this
    capture 14,877 of the 41,431 surviving points (36%) were observed by a
    withheld view, and their positions and colours carry that bundle adjustment.
    Deleting an observation afterwards does not undo its earlier influence - the
    kept points are bit-identical to the full model's.

A stricter protocol exists: reconstruct from the training images alone, then
localise the withheld cameras into that fixed model with COLMAP's image
registration. That is a different experiment and would need its own reported
number. What runs here is the standard 3DGS / Mip-NeRF-360 arrangement, and it
should be described as that rather than as full independence.

The baseline is the same question the mesh evaluation asks: can the reconstruction
beat simply showing the nearest photograph that was actually captured? A render
that cannot has not earned its reconstruction stage. Baseline candidates are
restricted to training views taken *with the same lens*, because this capture used
two (a 6.765 mm main and a 2.22 mm ultra-wide) and a cross-lens baseline would be
penalised for framing rather than for angle.

Training-view scores are also reported, unlabelled as a result, so the gap between
fit and generalisation is visible rather than implied.

``--object-radius`` adds a second, clearly secondary set of numbers measured only
inside the frog's own bounding box. Full-frame PSNR on this capture is dominated
by the room behind the frog - metres away, seen from a narrow range of angles, and
reconstructed as smear - which is not what the workshop is about. The box is the
projection of the sparse points lying within that radius of the scene median, so
it is defined by geometry rather than drawn by hand, and the identical box is
applied to the render, the withheld photograph and the baseline, so the comparison
stays symmetric. The full-frame number remains the result; this one is a diagnostic
that says how much of the error is the object and how much is the room.

    .venv-mlx3d/bin/python tools/eval_holdout.py \
        --splat model3d/gaussian/frog88_train/splat.ply \
        --model model3d/gaussian/frog88_undistorted \
        --split config/neural_split.csv --out output/neural
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import psnr, ssim  # noqa: E402


def to_bgr_u8(arr) -> np.ndarray:
    """MLX/NumPy float RGB in [0, 1] -> OpenCV uint8 BGR."""
    a = np.asarray(arr, dtype=np.float32)
    a = np.clip(a, 0.0, 1.0)
    return cv2.cvtColor((a * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


def sheet_crop(box, shape, pad_frac: float, aspect: float = 4 / 3):
    """A window around the object for the contact sheet, or None to keep the frame.

    Presentation only. The scores are computed on the whole photograph; this just
    decides what the sheet shows. Cropping matters because the frames are of a
    room in someone's home and the sheets are committed to a public repository,
    while the part that demonstrates the reconstruction is the frog. The same
    window is applied to all three columns, so the comparison stays aligned.
    """
    if box is None:
        return None
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = (x1 - x0) * (1 + 2 * pad_frac), (y1 - y0) * (1 + 2 * pad_frac)
    bw, bh = (max(bw, bh * aspect), max(bh, bw / aspect))
    bw, bh = min(bw, w), min(bh, h)
    bw, bh = (min(bw, bh * aspect), min(bh, bw / aspect))
    bw, bh = int(round(bw)), int(round(bh))
    cx0 = int(min(max(0, round(cx - bw / 2)), w - bw))
    cy0 = int(min(max(0, round(cy - bh / 2)), h - bh))
    return cx0, cy0, cx0 + bw, cy0 + bh


def ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    """Mean, half-width, t statistic and p-value for a paired difference."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, float("nan"), float("nan"), float("nan")
    sem = float(np.std(values, ddof=1) / math.sqrt(n))
    if sem == 0:
        # Every difference identical. A t statistic is undefined here, not
        # infinite, and reporting p=0 would announce certainty from a sample
        # that contains no variation at all. Say undefined and let the caller
        # print nan rather than a spurious significance.
        return mean, 0.0, float("nan"), float("nan")
    try:
        from scipy import stats

        crit = float(stats.t.ppf(0.975, n - 1))
        p = float(2 * stats.t.sf(abs(mean / sem), n - 1))
    except Exception:  # normal approximation if scipy is unavailable
        crit, p = 1.96, float("nan")
    return mean, crit * sem, mean / sem, p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splat", required=True, help="trained splat.ply")
    ap.add_argument("--model", required=True, help="COLMAP workspace holding ALL poses (sparse/0)")
    ap.add_argument("--images", default=None, help="image dir (default: <model>/images)")
    ap.add_argument("--split", default="config/neural_split.csv")
    ap.add_argument(
        "--train-model", default=None,
        help="sparse model the splat was actually trained on "
             "(default: <splat dir>/sparse/0). Checked against the split.",
    )
    ap.add_argument("--out", default="output/neural")
    ap.add_argument("--downscale", type=int, required=True, help="must match the training downscale")
    ap.add_argument("--contact-sheet", action="store_true")
    ap.add_argument(
        "--require-finite", action="store_true",
        help="fail instead of dropping Gaussians that hold non-finite values",
    )
    ap.add_argument(
        "--sheet-pad", type=float, default=0.45,
        help="context around the object in the contact sheet, as a fraction of the "
             "object box; 0 crops to the box itself. Presentation only - scores are "
             "always computed on the whole photograph",
    )
    ap.add_argument(
        "--object-radius", type=float, default=None,
        help="radius about the scene median bounding the object, in COLMAP units; "
             "enables the secondary object-region diagnostic",
    )
    args = ap.parse_args()

    import mlx.core as mx
    from mlx3d.datasets import load_colmap
    from mlx3d.splatting import GaussianModel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.split) as f:
        split = {r["name"]: r for r in csv.DictReader(f)}
    holdout = {n for n, r in split.items() if r["split"] == "holdout"}

    images_dir = args.images or str(Path(args.model) / "images")
    ds = load_colmap(args.model, images_dir=images_dir, downscale=args.downscale, cache="uint8")
    print(f"loaded {len(ds)} posed views from {args.model} at downscale {args.downscale}")

    # load_ply infers the SH degree from the file and sets active_sh_degree,
    # so the checkpoint renders at the degree it was trained with.
    model = GaussianModel.load_ply(args.splat)
    print(f"loaded splat: {model.num_gaussians:,} Gaussians, SH degree {model.active_sh_degree}")

    # MLX3D's compactor selects on opacity and importance; it has no finite-value
    # filter, so a training run that produced a degenerate Gaussian exports it.
    # Two of this project's three runs did: 3 rows in fast, 146 in MCMC, with
    # NaN in position, scale and rotation as well as the high-order SH bands.
    # A NaN position makes every comparison in the tile binner false, so such a
    # Gaussian is almost certainly invisible already - but "almost certainly" is
    # not a basis for a reported number. Drop them explicitly and record how many.
    n_before = model.num_gaussians
    finite = np.ones(n_before, dtype=bool)
    for key, value in model.params.items():
        arr = np.asarray(value).reshape(n_before, -1)
        finite &= np.isfinite(arr).all(axis=1)
    n_dropped = int((~finite).sum())
    if n_dropped:
        if args.require_finite:
            print(
                f"error: {n_dropped} of {n_before:,} Gaussians hold non-finite values. "
                "Re-run without --require-finite to drop and continue."
            )
            return 2
        model.select(np.where(finite)[0])
        print(
            f"dropped {n_dropped} non-finite Gaussian(s) before rendering "
            f"({100 * n_dropped / n_before:.4f}% of the checkpoint)"
        )

    missing = holdout - set(ds.image_names)
    if missing:
        print(f"error: {len(missing)} held-out view(s) absent from the model: {sorted(missing)[:5]}")
        return 2

    # Verify the split against the model the splat was ACTUALLY trained on, not
    # against whatever the CSV currently says. The CSV can be rewritten and a
    # previously-built quality workspace reused, and then a training image would
    # be reported as a held-out result -- the single worst failure this script
    # could have. Reading the trained model's own image list closes that off
    # regardless of how the workspaces were cached.
    train_sparse = Path(args.train_model or (Path(args.splat).parent / "sparse" / "0"))
    if not (train_sparse / "images.bin").exists():
        print(
            f"error: cannot find the training model at {train_sparse}. Pass --train-model. "
            "The split cannot be verified without it, and an unverified split must not be scored."
        )
        return 2
    from mlx3d.datasets.colmap import _read_images_bin

    trained_on = {m["name"] for m in _read_images_bin(str(train_sparse / "images.bin")).values()}
    leaked = sorted(trained_on & holdout)
    if leaked:
        print(
            f"error: {len(leaked)} view(s) marked held out were in the splat's training "
            f"model: {leaked[:5]}. The workspace is stale with respect to "
            f"{args.split}; rebuild it with tools/make_train_workspace.py."
        )
        return 2
    unaccounted = set(ds.image_names) - trained_on - holdout
    if unaccounted:
        print(
            f"error: {len(unaccounted)} view(s) are neither trained on nor held out: "
            f"{sorted(unaccounted)[:5]}. The split does not describe this splat."
        )
        return 2
    print(
        f"split verified against {train_sparse}: {len(trained_on)} trained, "
        f"{len(holdout)} held out, no overlap"
    )

    centers = np.stack([np.asarray(c.camera_center) for c in ds.cameras])
    obj = np.median(np.asarray(ds.points), axis=0)
    dirs = centers - obj
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    obj_pts = None
    if args.object_radius is not None:
        pts = np.asarray(ds.points)
        obj_pts = pts[np.linalg.norm(pts - obj, axis=1) < args.object_radius]
        print(
            f"object region: {len(obj_pts):,} of {len(pts):,} sparse points within "
            f"{args.object_radius} units of the scene median"
        )
        if len(obj_pts) < 50:
            print("error: too few points inside --object-radius to define a box")
            return 2

    def object_box(i: int, shape) -> tuple[int, int, int, int] | None:
        """Pixel box bounding the object in view i, or None if it is not visible.

        The 1st-99th percentile rather than the extremes, so a handful of stray
        points triangulated onto the object do not inflate the box.
        """
        cam = ds.cameras[i]
        xy, depth = cam.project_points(mx.array(obj_pts.astype(np.float32)))
        xy, depth = np.asarray(xy), np.asarray(depth)
        keep = (depth > 0) & np.isfinite(xy).all(axis=1)
        if keep.sum() < 50:
            return None
        xy = xy[keep]
        x0, y0 = np.percentile(xy, 1, axis=0)
        x1, y1 = np.percentile(xy, 99, axis=0)
        h, w = shape[:2]
        x0, y0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        x1, y1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
        if x1 - x0 < 16 or y1 - y0 < 16:  # SSIM needs an 11px window
            return None
        return x0, y0, x1, y1

    def render(i: int) -> np.ndarray:
        out_ = model.render(ds.cameras[i])
        mx.eval(out_["image"])
        return to_bgr_u8(out_["image"])

    rows = []
    sheet = []
    for i, name in enumerate(ds.image_names):
        gt = to_bgr_u8(ds.images[i])
        rendered = render(i)
        is_holdout = name in holdout
        rec = {
            "name": name,
            "ring": split.get(name, {}).get("ring", ""),
            "split": "holdout" if is_holdout else "train",
            "azimuth_deg": split.get(name, {}).get("azimuth_deg", ""),
            "elevation_deg": split.get(name, {}).get("elevation_deg", ""),
            "width": gt.shape[1],
            "height": gt.shape[0],
            "psnr_render": round(psnr(gt, rendered), 4),
            "ssim_render": round(ssim(gt, rendered), 5),
        }

        box = object_box(i, gt.shape) if obj_pts is not None else None
        if box is not None:
            x0, y0, x1, y1 = box
            rec["obj_box_frac"] = round((x1 - x0) * (y1 - y0) / (gt.shape[0] * gt.shape[1]), 5)
            rec["psnr_render_obj"] = round(psnr(gt[y0:y1, x0:x1], rendered[y0:y1, x0:x1]), 4)
            rec["ssim_render_obj"] = round(ssim(gt[y0:y1, x0:x1], rendered[y0:y1, x0:x1]), 5)

        if is_holdout:
            # Nearest *training* view shot with the same lens, by viewing angle.
            same_lens = [
                j
                for j, n in enumerate(ds.image_names)
                if n not in holdout
                and ds.cameras[j].width == ds.cameras[i].width
                and ds.cameras[j].height == ds.cameras[i].height
            ]
            cos = np.clip(dirs[same_lens] @ dirs[i], -1, 1)
            best = same_lens[int(np.argmax(cos))]
            sep = float(np.degrees(np.arccos(cos.max())))
            near = to_bgr_u8(ds.images[best])
            if near.shape != gt.shape:
                near = cv2.resize(near, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_AREA)
            rec.update(
                {
                    "baseline_name": ds.image_names[best],
                    "baseline_sep_deg": round(sep, 2),
                    "psnr_baseline": round(psnr(gt, near), 4),
                    "ssim_baseline": round(ssim(gt, near), 5),
                }
            )
            rec["d_psnr"] = round(rec["psnr_render"] - rec["psnr_baseline"], 4)
            rec["d_ssim"] = round(rec["ssim_render"] - rec["ssim_baseline"], 5)
            if box is not None:
                x0, y0, x1, y1 = box
                rec["psnr_baseline_obj"] = round(psnr(gt[y0:y1, x0:x1], near[y0:y1, x0:x1]), 4)
                rec["ssim_baseline_obj"] = round(ssim(gt[y0:y1, x0:x1], near[y0:y1, x0:x1]), 5)
                rec["d_psnr_obj"] = round(rec["psnr_render_obj"] - rec["psnr_baseline_obj"], 4)
                rec["d_ssim_obj"] = round(rec["ssim_render_obj"] - rec["ssim_baseline_obj"], 5)
            if args.contact_sheet:
                sheet.append((name, gt, rendered, near, rec, box))
            print(
                f"  holdout {name:<12} PSNR {rec['psnr_render']:6.2f} vs baseline "
                f"{rec['psnr_baseline']:6.2f} ({sep:4.1f} deg away)  "
                f"SSIM {rec['ssim_render']:.3f} vs {rec['ssim_baseline']:.3f}"
            )
        rows.append(rec)

    fields = [
        "name", "ring", "split", "azimuth_deg", "elevation_deg", "width", "height",
        "psnr_render", "ssim_render", "baseline_name", "baseline_sep_deg",
        "psnr_baseline", "ssim_baseline", "d_psnr", "d_ssim",
        "obj_box_frac", "psnr_render_obj", "ssim_render_obj",
        "psnr_baseline_obj", "ssim_baseline_obj", "d_psnr_obj", "d_ssim_obj",
    ]
    csv_path = out / "holdout_metrics.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    hold = [r for r in rows if r["split"] == "holdout"]
    train = [r for r in rows if r["split"] == "train"]
    dp = np.array([r["d_psnr"] for r in hold])
    dsm = np.array([r["d_ssim"] for r in hold])
    mp, hp, tp, pp = ci95(dp)
    ms, hs, ts, ps = ci95(dsm)

    lines = []
    lines.append(f"splat            {args.splat}")
    lines.append(f"gaussians        {model.num_gaussians:,}")
    if n_dropped:
        lines.append(
            f"non-finite       {n_dropped} Gaussian(s) dropped from the checkpoint before rendering"
        )
    lines.append(f"held-out views   {len(hold)} of {len(rows)}")
    lines.append(f"split verified   against {train_sparse} ({len(trained_on)} trained, no overlap)")
    lines.append("")
    lines.append("TRAINING FIT (not a result - the splat saw these pixels)")
    lines.append(
        f"  PSNR {np.mean([r['psnr_render'] for r in train]):.2f} dB   "
        f"SSIM {np.mean([r['ssim_render'] for r in train]):.4f}   n={len(train)}"
    )
    lines.append("")
    lines.append("HELD-OUT (novel view)")
    lines.append(
        f"  render     PSNR {np.mean([r['psnr_render'] for r in hold]):.2f} dB   "
        f"SSIM {np.mean([r['ssim_render'] for r in hold]):.4f}"
    )
    lines.append(
        f"  baseline   PSNR {np.mean([r['psnr_baseline'] for r in hold]):.2f} dB   "
        f"SSIM {np.mean([r['ssim_baseline'] for r in hold]):.4f}   "
        f"(nearest same-lens training photo, mean separation "
        f"{np.mean([r['baseline_sep_deg'] for r in hold]):.1f} deg)"
    )
    lines.append("")
    lines.append("PAIRED DIFFERENCE, render minus baseline (per held-out view)")
    lines.append(f"  dPSNR  {mp:+.3f} dB  95% CI [{mp - hp:+.3f}, {mp + hp:+.3f}]  n={len(dp)}  t={tp:.2f}  p={pp:.4f}")
    lines.append(f"  dSSIM  {ms:+.4f}     95% CI [{ms - hs:+.4f}, {ms + hs:+.4f}]  n={len(dsm)}  t={ts:.2f}  p={ps:.4f}")
    lines.append("")
    for metric, mean, half in (("PSNR", mp, hp), ("SSIM", ms, hs)):
        verdict = (
            "significant" if (mean - half) * (mean + half) > 0 else "NOT significant (interval spans zero)"
        )
        lines.append(f"  {metric}: {verdict}")
    if obj_pts is not None and all("d_psnr_obj" in r for r in hold):
        dpo = np.array([r["d_psnr_obj"] for r in hold])
        dso = np.array([r["d_ssim_obj"] for r in hold])
        mpo, hpo, tpo, ppo = ci95(dpo)
        mso, hso, tso, pso = ci95(dso)
        frac = np.mean([r["obj_box_frac"] for r in hold])
        lines.append("")
        lines.append(
            f"SECONDARY DIAGNOSTIC - object region only (radius {args.object_radius} "
            f"units, mean {100 * frac:.1f}% of frame)"
        )
        lines.append("  This is NOT the headline result. It isolates how much of the")
        lines.append("  full-frame error is the frog and how much is the room behind it.")
        lines.append(
            f"  render     PSNR {np.mean([r['psnr_render_obj'] for r in hold]):.2f} dB   "
            f"SSIM {np.mean([r['ssim_render_obj'] for r in hold]):.4f}"
        )
        lines.append(
            f"  baseline   PSNR {np.mean([r['psnr_baseline_obj'] for r in hold]):.2f} dB   "
            f"SSIM {np.mean([r['ssim_baseline_obj'] for r in hold]):.4f}"
        )
        lines.append(f"  dPSNR  {mpo:+.3f} dB  95% CI [{mpo - hpo:+.3f}, {mpo + hpo:+.3f}]  t={tpo:.2f}  p={ppo:.4f}")
        lines.append(f"  dSSIM  {mso:+.4f}     95% CI [{mso - hso:+.4f}, {mso + hso:+.4f}]  t={tso:.2f}  p={pso:.4f}")

    lines.append("")
    lines.append("per-ring held-out means")
    for ring in sorted({r["ring"] for r in hold}):
        sub = [r for r in hold if r["ring"] == ring]
        lines.append(
            f"  {ring:<5} n={len(sub)}  render PSNR {np.mean([r['psnr_render'] for r in sub]):6.2f}  "
            f"baseline {np.mean([r['psnr_baseline'] for r in sub]):6.2f}  "
            f"dSSIM {np.mean([r['d_ssim'] for r in sub]):+.4f}"
        )

    summary = "\n".join(lines)
    print("\n" + summary)
    (out / "holdout_summary.txt").write_text(summary + "\n")
    print(f"\nwrote {csv_path}")
    print(f"wrote {out / 'holdout_summary.txt'}")

    if sheet:
        tiles = []
        for name, gt, rendered, near, rec, box in sheet:
            h = 240
            crop = sheet_crop(box, gt.shape, args.sheet_pad)
            cx0, cy0, cx1, cy1 = crop if crop else (0, 0, gt.shape[1], gt.shape[0])

            def fit(img, label, box=box):
                view = img[cy0:cy1, cx0:cx1]
                s_ = h / view.shape[0]
                scaled = cv2.resize(view, (max(1, int(view.shape[1] * s_)), h))
                if box is not None:
                    # The region the secondary diagnostic scores, in crop coords, so
                    # the reader can see which pixels that number is about.
                    x0, y0, x1, y1 = (
                        int(round((box[0] - cx0) * s_)), int(round((box[1] - cy0) * s_)),
                        int(round((box[2] - cx0) * s_)), int(round((box[3] - cy0) * s_)),
                    )
                    cv2.rectangle(scaled, (x0, y0), (x1, y1), (0, 235, 255), 1)
                cv2.rectangle(scaled, (0, 0), (scaled.shape[1] - 1, 18), (0, 0, 0), -1)
                cv2.putText(scaled, label, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
                return scaled
            tiles.append(
                np.hstack([
                    fit(gt, f"{name} withheld photo"),
                    fit(rendered, f"splat @ same pose  full {rec['psnr_render']:.1f}dB"
                        + (f" | obj {rec['psnr_render_obj']:.1f}dB" if 'psnr_render_obj' in rec else "")),
                    fit(near, f"baseline {rec['baseline_name']} {rec['baseline_sep_deg']:.0f}deg  full {rec['psnr_baseline']:.1f}dB"
                        + (f" | obj {rec['psnr_baseline_obj']:.1f}dB" if 'psnr_baseline_obj' in rec else "")),
                ])
            )
        width = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 0), (0, width - t.shape[1]), (0, 0))) for t in tiles]
        banner = np.zeros((22, width, 3), np.uint8)
        cv2.putText(
            banner,
            "cropped to the object; dB labels are FULL-FRAME scores over the whole photograph",
            (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 235, 255), 1, cv2.LINE_AA,
        )
        path = out / "holdout_contact_sheet.jpg"
        cv2.imwrite(str(path), np.vstack([banner] + tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
