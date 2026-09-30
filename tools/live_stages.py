"""The staged run through the API (tools/live_stages.py [space] [preset]): every intermediate output the page would show
(the stage names with their times) and the final headline, as a streaming job."""
import os, sys, time
from gradio_client import Client
from huggingface_hub import get_token

space = sys.argv[1] if len(sys.argv) > 1 else "rfick/disco"
preset = sys.argv[2] if len(sys.argv) > 2 else "clinical b1000 x 30"
c = Client(space, token=get_token(), verbose=False, download_files=os.environ.get("LIVE_OUT", "/tmp/disco-live"), httpx_kwargs={"timeout": 900})
physics = [True, 3.0, "along z (the strands' frame)", 0, 0, 50.0, 55.0, 10.0, 1200.0, 1000.0, 440.0, 1.16, -0.1, -0.1, True, True, True]
shells = [True, "d12-D24", 1000, 30, 12.0, 24.0, 53.5, True, "d8-D20", 3000, 45, 8.0, 20.0, 53.5,
          False, "d17-D30", 3000, 90, 17.0, 30.0, 53.5, False, "d17-D30", 6000, 60, 17.0, 30.0, 53.5]
t0 = time.perf_counter()
job = c.submit(preset, 2, True, 30, 2, 30.0, 0.5, 0, None, *physics, *shells, api_name="/run_pipeline")
seen = 0
while not job.done():
    outs = job.outputs()
    for o in outs[seen:]:
        print(f"  +{time.perf_counter() - t0:5.1f} s: {str(o[1])[:100]}", flush=True)
    seen = len(outs)
    time.sleep(1)
outs = job.outputs()
for o in outs[seen:]:
    print(f"  +{time.perf_counter() - t0:5.1f} s: {str(o[1])[:100]}", flush=True)
print(f"{preset}: {len(outs)} updates, {time.perf_counter() - t0:.1f} s wall", flush=True)
