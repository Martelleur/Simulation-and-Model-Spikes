from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def run_consumer(q, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)
    states_path = out / "states.csv"

    # CSV headers
    with open(states_path, "w") as f:
        f.write("t,rx,ry,rz,vx,vy,vz,alt\n")

    frames = 0
    last_alt = None
    while True:
        msg = q.get()
        if msg.get("type") == "END":
            break
        typ = msg.get("type")
        if typ == "state":
            r = msg["r"]; v = msg["v"]
            with open(states_path, "a") as f:
                f.write(f"{msg['t']},{r[0]},{r[1]},{r[2]},{v[0]},{v[1]},{v[2]},{msg['alt']}\n")
            last_alt = msg["alt"]
        elif typ == "frame":
            img = Image.fromarray(msg["img"], mode="L")
            img.save(frames_dir / f"frame_{msg['id']:06d}.png")
            frames += 1
        elif typ == "summary":
            # write summary at end
            (out / "summary.json").write_text(__import__("json").dumps(msg, indent=2))

    # also write a small footer summary
    summary2 = { "frames_written": frames, "last_alt": last_alt }
    (out / "summary_extra.json").write_text(__import__("json").dumps(summary2, indent=2))
