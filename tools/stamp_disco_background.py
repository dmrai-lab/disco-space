"""Stamp the background moments onto DiSCo's shape-moment layout (dmipy-sim's shape_moments.stamp_background): one
pass over the columnar source, the bg_<g>.npy columns and the manifest's background and RF schedules written into
DIR, which holds the layout's manifest.json and tiles.npy (the rest of the layout is not read and not touched; copy
the new columns and manifest beside it). Measured 2026-10-03 on gaia's CPU from the Hub: 42.4 GB read, 1,016 s.

    python tools/stamp_disco_background.py DIR
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dmipy_sim as d
from dmipy_sim.replay.shape_moments import stamp_background
out = sys.argv[1]
man = json.load(open(os.path.join(out, "manifest.json")))
shapes = {}
for name, rec in man["shapes"].items():
    fam, kw = rec["build_spec"]
    kw = dict(kw); dirs = kw.pop("gradient_directions")
    if fam == "pgse":
        shapes[name] = d.pgse(dirs, kw["delta"], kw["Delta"], gradient_strengths=kw["gradient_strengths"], TE=kw["TE"], n_t=kw["n_t"], slew_rate=np.inf)
    else:
        shapes[name] = d.pgste(dirs, kw["delta"], kw["TM"], gradient_strengths=kw["gradient_strengths"], n_t=kw["n_t"], slew_rate=np.inf,
                               ste_flip_angles=tuple(kw["ste_flip_angles"]))
last = [0.0]
def progress(rows, nbytes, secs):
    if secs - last[0] > 60:
        last[0] = secs
        print(time.strftime("%H:%M:%S"), f"{rows:,} rows, {nbytes / 1e9:.1f} GB, {secs:.0f} s, {nbytes / max(secs, 1) / 1e6:.0f} MB/s", flush=True)
m = stamp_background(out, "hf://SubstrateCommons/disco-replay/disco", shapes, workers=16, progress=progress)
print("DONE", json.dumps(m["background"]), json.dumps(m["stamps"][-1]), flush=True)
