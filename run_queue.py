"""Sequential arm queue, VRAM-gated.

Waits until nvidia-smi reports at least FREE_NEED MiB free (so the queue never
disturbs timing-sensitive jobs already on the card), then runs each arm.
Resume-safe: arms already present in results.jsonl are skipped.

Usage: python run_queue.py <free_mib_required> <arm1> <arm2> ...
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FREE_NEED = int(sys.argv[1])
ARMS = sys.argv[2:]


def free_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return int(out.decode().strip().splitlines()[0])


print(f"queue: waiting for {FREE_NEED} MiB free", flush=True)
waited = 0
while free_mib() < FREE_NEED:
    waited += 30
    if waited % 300 == 0:
        print(f"queue: still waiting ({free_mib()} MiB free)", flush=True)
    time.sleep(30)
print(f"queue: enough free VRAM after {waited}s - starting", flush=True)

results = os.path.join(HERE, "results", "results.jsonl")
done = set()
if os.path.exists(results):
    done = {json.loads(l)["run"] for l in open(results)}

for arm in ARMS:
    if arm in done:
        print(f"queue: {arm} already done, skip", flush=True)
        continue
    print(f"queue: START {arm} {time.strftime('%H:%M:%S')}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "harness.py"), arm],
                       cwd=HERE)
    print(f"queue: {'DONE' if r.returncode == 0 else f'FAILED rc={r.returncode}'} "
          f"{arm} {time.strftime('%H:%M:%S')}", flush=True)
print("queue: ALL DONE", flush=True)
