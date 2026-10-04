"""What a machine of the scanner menu costs on the brain: the scan's own protocol at the default panel on the ideal
scanner and on each machine of ``brain.toml``'s menu -- the encoding classes, the packs' pose responses (``prepare``,
JAX on the CPU as in the page's process and the GPU worker) and the contraction on the device (``replay``, first call
and steady) -- and the delivered image's departure from the ideal one. Writes a JSON (docs/scanner.md quotes it).

    python tools/measure_scanners.py --asset DIR --wm-pack WM.rpk --gm-pack GM.rpk [--out brain_scanners.json]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from space import pipeline as P            # noqa: E402
from space.sources import brain as B       # noqa: E402


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True)
    ap.add_argument("--wm-pack", required=True)
    ap.add_argument("--gm-pack", required=True)
    ap.add_argument("--out", default="brain_scanners.json")
    a = ap.parse_args()
    cfg = P.config(os.path.join(P.HERE, "brain.toml"))
    cfg["asset"] = {"local": a.asset}
    cfg["packs"] = {"wm": [{"label": "wm", "uri": a.wm_pack}], "gm": [{"label": "gm", "uri": a.gm_pack}]}
    src = B.Brain(cfg)
    panel = B.Brain.panel(cfg)
    v = {c.name: c.value for row in panel.rows for c in row if c.name in panel.fields}
    v.update(wm_pack="wm", gm_pack="gm")
    prot = B.Brain.protocol(cfg, B.Brain.presets(cfg)[0])
    out = {}

    def measure(ph):
        meas = src.validate(prot, ph)
        t0 = time.perf_counter(); n_cls = src.n_classes(meas, ph); t_cls = time.perf_counter() - t0
        t0 = time.perf_counter(); k = src.prepare(meas, ph); t_prep = time.perf_counter() - t0
        secs = []
        for _ in range(2):
            sync(); t0 = time.perf_counter(); dwi, _, factor, _ = src.replay(meas, ph, k); sync(); secs.append(time.perf_counter() - t0)
        row = dict(scanner=ph.scanner, field_T=ph.field_T, n_meas=int(len(meas.bvals)), encoding_classes=int(n_cls), classes_seconds=t_cls,
                   prepare_seconds=t_prep, prepare_stages=k.stages, replay_first=secs[0], replay_steady=secs[1])
        return row, dwi * factor[..., None]

    for label, key in P.machines(cfg).items():
        L = P.limits(key)
        on = B.Brain.physics_from(cfg, v, scanner=key)
        priced = src.responses([(src.validate(prot, on), on)])                 # before prepare caches it
        ideal, S0 = measure(B.Brain.physics_from(cfg, dict(v, field_T=float(L.field_T), b0_mode=list(B.B0_MODES)[0])))
        row, S = measure(on)
        both = np.isfinite(S0) & np.isfinite(S)
        d = np.abs(S - S0)[both]
        row.update(ideal_at_its_field=ideal, max_abs_dS=float(d.max()), median_abs_dS=float(np.median(d)), priced=priced,
                   reserved=B.Brain.estimated_seconds(cfg, prot, density=cfg["tracking"]["density"], knob=P.NO_KNOB, n_keys=1, ladder=True,
                                                      responses=priced, scanner=key))
        out[label] = row
        print(label, json.dumps(row, indent=1, default=str), flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=str)


if __name__ == "__main__":
    main()
