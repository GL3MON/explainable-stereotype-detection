"""Poll a submitted Kaggle job and retrieve its final training artifacts."""

import argparse
import contextlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_directory", type=Path)
    args = parser.parse_args()
    root = args.job_directory.resolve()
    job = json.loads((root / "job.json").read_text())
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        import kaggle
    api = kaggle.api
    while True:
        try:
            response = api.kernels_status(job["kernel"])
            state = response.status.name
            payload = {"state": state, "checked": time.time(), "failure_message": response.failure_message,
                       "url": job["url"]}
            (root / "monitor_status.json").write_text(json.dumps(payload, indent=2) + "\n")
            try:
                log = api.kernels_logs(job["kernel"])
                (root / "remote.log").write_text(log)
            except Exception as error:
                print("Log retrieval:", type(error).__name__, flush=True)
            print(json.dumps(payload), flush=True)
            if state in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
                downloads = root / "download"
                files, _ = api.kernels_output(job["kernel"], str(downloads),
                                             file_pattern=r"(training_artifacts\.zip|job_status\.json)$",
                                             quiet=True)
                archive = downloads / "training_artifacts.zip"
                if archive.exists():
                    results = root / "results"
                    results.mkdir(exist_ok=True)
                    with zipfile.ZipFile(archive) as z:
                        assert z.testzip() is None, "Downloaded ZIP failed CRC validation"
                        for entry in z.infolist():
                            assert (results / entry.filename).resolve().is_relative_to(results)
                        z.extractall(results)
                    payload["artifacts_downloaded"] = True
                    payload["results_directory"] = str(results)
                else:
                    payload["artifacts_downloaded"] = False
                payload["downloaded_files"] = files
                (root / "monitor_status.json").write_text(json.dumps(payload, indent=2) + "\n")
                print("RETRIEVAL", json.dumps(payload), flush=True)
                return
        except Exception as error:
            print("Monitor request:", type(error).__name__, flush=True)
        time.sleep(45)


if __name__ == "__main__":
    main()
