"""The live Space through its Gradio API (tools/live_check.py [space]): the DiSCo 364 preset at SNR 30, then a custom
two-shell protocol; prints the headline and timings, downloads the figures and files to $LIVE_OUT."""
import os, sys, time
from gradio_client import Client
from huggingface_hub import get_token

space = sys.argv[1] if len(sys.argv) > 1 else "rfick/disco"
c = Client(space, token=get_token(), verbose=False, download_files=os.environ.get("LIVE_OUT", "/tmp/disco-live"))
shells = [True, "d12-D24", 1000, 30, True, "d8-D20", 3000, 45, False, "d17-D30", 3000, 90, False, "d17-D30", 6000, 60]
for preset in ("DiSCo 364", "custom shells"):
    t0 = time.perf_counter()
    out = c.predict(preset, 2, True, 30, 4, 30.0, 0.5, 0, None, None, "d12-D24", *shells, api_name="/run_pipeline")
    print(preset, f"{time.perf_counter() - t0:.1f} s wall", flush=True)
    headline, dwi, tract, mats, timings, tck, vols = out
    print(headline)
    print(timings)
    print("figures", dwi, tract, mats, "tck", tck, "volumes", vols, flush=True)
