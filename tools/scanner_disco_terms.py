"""Each term alone on the DiSCo layout: the Swoop at 7.9 cm along R-L, the Swoop-playable shell (b 350 x 60 on
d17.7-D35.8), noiseless, the catalogue's white matter at 64 mT: the DWI's change against the ideal scanner per term
(the S0-normalised signal, so the transmit scale's common factor cancels and only its S0 survives).

    LAYOUT=/stamped/layout python tools/scanner_disco_terms.py
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from space import pipeline as P
from space.sources import disco as D
from dmipy_sim.replay.shape_moments import ShapeMoments
from dmipy_tract import hemisphere
sm = ShapeMoments(os.environ["LAYOUT"])
L = P.limits("hyperfine_swoop_64mT")
ph = P.Physics.at(0.064, pools=("intra", "extra"), b0_direction=(0.0, 1.0, 0.0))
tissue, scanner = P.tissue_and_scanner(ph, ("myelin",))
dirs = np.concatenate([np.tile([[0.0, 0.0, 1.0]], (3, 1)), hemisphere(60)]); b = np.r_[np.zeros(3), np.full(60, 350e6)]
grid = sm.grid.centred_at((0.079, 0.0, 0.0))
def img(**terms):
    dl = None if not terms else sm.delivery("d17.7-D35.8", b, dirs, L, grid, **terms)
    S, fl = sm.image("d17.7-D35.8", b, dirs, tissue=tissue, scanner=scanner, b0_direction=(0.0, 1.0, 0.0), delivered=dl)
    return S / np.nanmean(S[..., :3], -1, keepdims=True), fl, S[..., :3].mean(-1)
off = dict(nonlinearity=False, background=False, concomitant=False, transmit=False)
A, floor, S0A = img()
m = np.isfinite(A[..., 0])
out = {}
for term in ("nonlinearity", "background", "concomitant", "transmit", "all"):
    Bt, _, S0B = img(**({k: True for k in off} if term == "all" else {**off, term: True}))
    d = np.abs(Bt - A)[m][:, 3:]
    mean = (Bt[m][:, 3:].mean(1) - A[m][:, 3:].mean(1)) / A[m][:, 3:].mean(1)
    out[term] = dict(abs_dwi_median=float(np.median(d)), abs_dwi_p99=float(np.quantile(d, 0.99)), direction_mean_rel_median=float(np.median(mean)),
                     s0_ratio_median=float(np.median(S0B[m] / S0A[m])))
    print(term, json.dumps(out[term]), flush=True)
out["floor_median"] = float(np.nanmedian(floor[m]))
json.dump(out, open("disco_terms.json", "w"), indent=1)
