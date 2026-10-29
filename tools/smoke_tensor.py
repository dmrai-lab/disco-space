"""The b-tensor preset through the page's own run button without a browser: A at 3 T with B = A with the field off
(the knob), the explorer's ladder on; every output of :data:`space.app.OUTPUTS` is produced and the b-tensor ones are
images; then a voxel click, an STE row of the DWI slice and the explorer's B − A of µFA.

    DISCO_LAYOUT=/local/layout python tools/smoke_tensor.py [--snr 30]
"""
import argparse
import json
import time
import types

import numpy as np


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--snr", type=float, default=30.0); ap.add_argument("--knob", default="field tier → off")
    a = ap.parse_args()
    from space import app as A
    from space import pipeline as P
    st = A._load()
    if st["error"]:
        raise SystemExit(st["error"])
    src, cfg = st["source"], st["cfg"]; S = type(src)
    preset = next(n for n, v in cfg["presets"].items() if "tensor" in v)
    panel = S.panel(cfg)
    vals = {c.name: c.value for row in panel.rows for c in row}
    values = [vals[f] for f in panel.fields]
    shells = [False, "d12-D24", 1000, 30, 10.0, 20.0, 60.0] * A.MAX_SHELLS
    args = (preset, 1, True, a.snr, 2, 30.0, 0.5, 0, None, a.knob, A.IDEAL, 1, True, *values, *shells)
    t0 = time.perf_counter(); last = None; texts = []
    for item in A.run_pipeline(A.compute, *args):
        last = item
        if isinstance(item, tuple) and isinstance(item[1], str):
            texts.append(item[1])
    out = dict(zip(A.OUTPUTS, last)); secs = time.perf_counter() - t0
    rs = out["result"]
    kinds = {k: type(v).__name__ for k, v in out.items()}
    for k in ("tensor_view", "tensor_pick", "tensor_view_b", "dwi_view", "explore_view", "metric_view"):
        assert out[k] is not None and hasattr(out[k], "size"), (k, kinds[k])
    assert out["tract_view"] is None and out["mats"] is None
    evt = types.SimpleNamespace(index=(20 * A.PICK + 3, 19 * A.PICK + 3))
    curve = A.pick_voxel(rs, 20, evt)
    ste = A.redraw_slice(rs, 20, 3, True, "A", "STE")
    ex = A.redraw_explorer(rs, rs["explorer"]["layers"][-1][0], "B − A", A.METRICS[3], 20, 0, "LTE")
    tm = rs["A"]["tensor"]; sig = np.isfinite(rs["A"]["dwi"][..., 0])
    report = dict(seconds=secs, headline=out["headline"][:600], stages=texts[-6:], outputs=kinds,
                  ufa_median=float(np.nanmedian(tm["ufa"][sig])), fa_median=float(np.nanmedian(tm["fa"][sig])),
                  md_median=float(np.nanmedian(tm["md"][sig])), voxel_curve=curve is not None, ste_slice=ste[0] is not None,
                  explorer_b_minus_a_mufa=ex[0] is not None, files=[str(f).rsplit("/", 1)[-1] for f in out["volumes"]],
                  timings=out["timings"])
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
