"""Bake the workshop station into one self-contained HTML file.

`workshop/index.html` loads its frames from `workshop/frames/`, which is the
maintainable arrangement and works over HTTP and from `file://`. It also depends
on the folder travelling with the file. On the day, the station gets opened from
whatever is to hand - a memory stick, a download, a copy in another directory -
and a page that renders an empty box because a sibling folder went missing is a
bad thing to discover in front of a room.

So this writes a second copy with every frame inlined as a data: URI and the
measurements already in the script. One file, no server, no network, nothing
beside it. Roughly 4 MB, which mail and any browser handle.

    .venv-depth/bin/python tools/build_workshop.py
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "workshop" / "index.html"
OUT = ROOT / "workshop" / "frog-station-standalone.html"
FRAMES = 36
SETS = {"relief": "relief", "mesh": "mesh", "splat": "splat"}


def main() -> int:
    html = SRC.read_text()
    embed, total = {}, 0
    for key, folder in SETS.items():
        uris = []
        for i in range(FRAMES):
            f = ROOT / "workshop" / "frames" / folder / f"f_{i:03d}.jpg"
            if not f.exists():
                print(f"error: missing {f}. Run ./run_workshop.sh frames first.")
                return 2
            raw = f.read_bytes()
            total += len(raw)
            uris.append("data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii"))
        embed[key] = uris

    marker = "<script>\n(function(){"
    if marker not in html:
        print("error: could not find the script block to inject into")
        return 2
    injected = (
        "<script>\nvar EMBED = "
        + json.dumps(embed, separators=(",", ":"))
        + ";\n</script>\n"
        + marker
    )
    html = html.replace(marker, injected, 1)
    html = html.replace(
        "<title>How many photographs make a frog?",
        "<title>How many photographs make a frog?",
    )
    OUT.write_text(html)
    print(f"embedded {FRAMES * len(SETS)} frames ({total / 1e6:.1f} MB of JPEG)")
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
