"""Persist the Kaggle live execution stream without printing credentials."""

import contextlib
import io
import json
import logging
import sys
import time
from pathlib import Path

logging.disable(logging.CRITICAL)
root = Path(sys.argv[1])
kernel = json.loads((root / "job.json").read_text())["kernel"]
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import kaggle
seen = set()
while True:
    try:
        with (root / "live.log").open("a") as out:
            for event in kaggle.api.kernels_logs_stream(kernel):
                key = (event.get("stream_name"), event.get("time"), event.get("data"))
                if key not in seen:
                    seen.add(key)
                    out.write(event.get("data", ""))
                    out.flush()
        if kaggle.api.kernels_status(kernel).status.name in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            break
    except Exception as error:
        print(type(error).__name__, flush=True)
    time.sleep(15)
