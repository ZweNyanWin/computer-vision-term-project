"""Verify the workshop station is complete, self-consistent and offline.

Written because the station will be opened once, in front of a room, from
whatever machine is to hand. The failure that matters is not a crash - it is the
station quietly showing a stale number, or an empty box because a folder did not
travel with the file. Both are silent, and both are checkable.

Checks, in the order they would bite:

  1. Every frame exists, decodes, and is the size the page expects.
  2. The counts inlined in index.html match workshop/frames/visibility.json.
     They are written from one source but live in two places, so they can drift.
  3. The standalone build carries all 108 frames as data: URIs and refers to no
     sibling file, so it works from a memory stick.
  4. Neither page reaches the network. A venue's wifi must not be able to change
     what the station shows.
  5. Every measured figure quoted in the prose still matches
     output/neural/balanced/holdout_summary.txt. Re-running the evaluation must
     not leave the station quoting last week's result.

    .venv-depth/bin/python tools/check_workshop.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / "workshop"
FRAMES = 36
SETS = ("relief", "mesh", "splat")

problems: list[str] = []
notes: list[str] = []


def bad(msg: str) -> None:
    problems.append(msg)


def main() -> int:
    index = W / "index.html"
    standalone = W / "frog-station-standalone.html"
    for p in (index, standalone):
        if not p.exists():
            bad(f"missing {p.relative_to(ROOT)} - run ./run_workshop.sh")
    if problems:
        return report()

    html = index.read_text()
    solo = standalone.read_text()

    # 1. frames -------------------------------------------------------------
    sizes = set()
    splat_black = []
    splat_edge_black = []
    for tag in SETS:
        got = sorted((W / "frames" / tag).glob("f_*.jpg"))
        if len(got) != FRAMES:
            bad(f"{tag}: {len(got)} frames, expected {FRAMES}")
            continue
        for f in got:
            im = cv2.imread(str(f))
            if im is None:
                bad(f"{f.relative_to(ROOT)} does not decode")
            else:
                sizes.add(im.shape[:2])
                if tag == "splat":
                    black = np.all(im < 12, axis=2)
                    edge = np.concatenate([
                        black[:12].ravel(), black[-12:].ravel(),
                        black[:, :12].ravel(), black[:, -12:].ravel(),
                    ])
                    splat_black.append(float(black.mean()))
                    splat_edge_black.append(float(edge.mean()))
    if len(sizes) > 1:
        bad(f"frames are not all the same size: {sorted(sizes)}")
    elif sizes:
        height, width = sizes.pop()
        notes.append(f"108 frames, all {width}x{height}")
    if splat_black:
        if min(splat_black) < 0.55 or min(splat_edge_black) < 0.98:
            bad("splat frames are not isolated on black - room/support pixels "
                "have leaked back into the workshop turntable")
        else:
            notes.append("splat foreground is isolated on black in every frame")

    # 1b. the turntable is a true cycle, not a sweep with a duplicated end -----
    # render3d.py spreads --frames across --sweep inclusively, so asking for 36
    # frames over 360 degrees ends where it started: a duplicate at the wrap and
    # a 10.2857-degree step that no simple formula reproduces, which silently
    # decouples the visible-surface counts from the frames they label. The mesh
    # and splat are detailed enough that a repeat is unmistakable in the pixels,
    # so check both. The relief is exempt: from behind it is legitimately almost
    # empty, and several of its frames are near-identical for that reason - which
    # is the exhibit, not a defect.
    for tag in ("mesh", "splat"):
        orbit_frames = sorted((W / "frames" / tag).glob("f_*.jpg"))
        if len(orbit_frames) == FRAMES:
            ims = [cv2.imread(str(f)).astype(np.int16) for f in orbit_frames]
            diffs = [
                float(np.abs(ims[i] - ims[(i + 1) % FRAMES]).mean())
                for i in range(FRAMES)
            ]
            repeats = [i for i, d in enumerate(diffs) if d < 0.5]
            if repeats:
                bad(f"{tag} frames {repeats} are identical to their neighbour - "
                    f"the turntable duplicates a frame at the wrap")
            else:
                notes.append(f"{tag} turntable is a true 36-step cycle "
                             f"(smallest neighbour difference {min(diffs):.1f})")

    # 2. inlined counts match the measured file ------------------------------
    vis_path = W / "frames" / "visibility.json"
    if not vis_path.exists():
        bad("missing workshop/frames/visibility.json - run ./run_workshop.sh frames")
    else:
        measured = json.loads(vis_path.read_text())
        m = re.search(r"var vis = (\{.*?\});", html, re.S)
        if not m:
            bad("index.html has no inlined `var vis = {...};`")
        else:
            inlined = json.loads(m.group(1))
            if inlined != measured:
                bad("the counts inlined in index.html differ from visibility.json "
                    "- re-run tools/measure_visibility.py")
            else:
                notes.append("inlined counts match visibility.json")
        for tag, body in measured.items():
            if len(body["visible"]) != FRAMES:
                bad(f"{tag}: {len(body['visible'])} counts, expected {FRAMES}")
            if max(body["visible"]) > body["total_triangles"]:
                bad(f"{tag}: a visible count exceeds the triangle total")

    # 3. standalone is genuinely standalone ----------------------------------
    n_data = solo.count("data:image/jpeg;base64,")
    if n_data != FRAMES * len(SETS):
        bad(f"standalone carries {n_data} embedded frames, expected {FRAMES * len(SETS)}")
    else:
        notes.append(f"standalone embeds all {n_data} frames")
    if re.search(r'"frames/[a-z]+/f_', solo.replace('? EMBED[k][i]', '')):
        # the fallback path string is allowed to remain, but it must never be
        # the thing that runs: EMBED is defined, so guard on that instead.
        if "var EMBED = {" not in solo:
            bad("standalone references frames/ but does not define EMBED")
    if "var EMBED = {" not in solo:
        bad("standalone does not define EMBED - it would need the frames folder")

    # 4. nothing reaches the network ----------------------------------------
    for name, text in (("index.html", html), ("standalone", solo)):
        for pat, what in ((r"https?://[^\s\"'<>)]+", "an absolute URL"),
                          (r"\bfetch\s*\(", "a fetch() call"),
                          (r"XMLHttpRequest", "an XMLHttpRequest"),
                          (r"<script[^>]+src=", "an external script")):
            hit = re.search(pat, text)
            if hit:
                bad(f"{name} contains {what}: {hit.group(0)[:60]}")
    if not any("network" in p for p in problems):
        notes.append("no network access in either page")

    # 5. quoted results still match the evaluation ---------------------------
    summary = ROOT / "output" / "neural" / "balanced" / "holdout_summary.txt"
    if not summary.exists():
        bad("missing output/neural/balanced/holdout_summary.txt")
    else:
        s = summary.read_text()

        def grab(pattern: str) -> str | None:
            m = re.search(pattern, s)
            return m.group(1) if m else None

        # Compare numerically, not as strings: the page rounds for readability
        # (+6.25 dB against the summary's 6.248) and should keep doing so. What
        # must not happen is the two disagreeing by more than that rounding.
        checks = [
            ("render PSNR", grab(r"render\s+PSNR\s+([\d.]+) dB"),
             r'class="num win">([\d.]+) dB<'),
            ("render SSIM", grab(r"render\s+PSNR[^\n]*SSIM\s+([\d.]+)"),
             r'class="num win">([\d.]+)<'),
            ("baseline PSNR", grab(r"baseline\s+PSNR\s+([\d.]+) dB"),
             r'class="num">([\d.]+) dB<'),
            ("baseline SSIM", grab(r"baseline\s+PSNR[^\n]*SSIM\s+([\d.]+)"),
             r'class="num">([\d.]+)<'),
            ("dPSNR", grab(r"dPSNR\s+\+([\d.]+) dB"),
             r'class="stat">\+([\d.]+) dB<'),
        ]
        for label, truth, pattern in checks:
            if truth is None:
                bad(f"could not read {label} from holdout_summary.txt")
                continue
            m = re.search(pattern, html)
            if not m:
                bad(f"index.html does not display {label} where expected")
                continue
            shown, actual = m.group(1), truth
            # tolerance = half a unit of the last digit the page shows
            dp = len(shown.split(".")[1]) if "." in shown else 0
            if abs(float(shown) - float(actual)) > 0.5 * 10 ** -dp:
                bad(f"index.html shows {label} as {shown}, but the evaluation says "
                    f"{actual} - re-run ./run_workshop.sh after re-evaluating")
        gauss = grab(r"gaussians\s+([\d,]+)")
        if gauss and gauss not in html:
            bad(f"index.html does not quote the current Gaussian count ({gauss})")
        ci = re.search(r"dPSNR[^\n]*95% CI \[\+([\d.]+), \+([\d.]+)\]", s)
        if ci and (ci.group(1)[:4] not in html or ci.group(2)[:4] not in html):
            bad(f"index.html does not quote the current confidence interval "
                f"[+{ci.group(1)}, +{ci.group(2)}]")
        if not any(("shows" in p or "quote" in p or "display" in p) for p in problems):
            notes.append("every quoted figure agrees with output/neural/balanced/")

    return report()


def report() -> int:
    for n in notes:
        print(f"  ok    {n}")
    if problems:
        print(f"\nNOT READY - {len(problems)} problem(s):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        return 1
    print("\nREADY - the station is complete, self-consistent and offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
