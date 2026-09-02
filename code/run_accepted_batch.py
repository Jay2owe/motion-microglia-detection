from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading
import time


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-prefix", default="r02_accepted_")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--stems", nargs="+", required=True)
    parser.add_argument("--resume-after-m5-prefix")
    parser.add_argument("--accepted-only-prefix")
    args = parser.parse_args()
    if args.resume_after_m5_prefix and args.accepted_only_prefix:
        parser.error("choose only one resume mode")

    project = Path(__file__).resolve().parents[1]
    config = args.config.resolve()
    log_root = args.log_root.resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    status_path = log_root / "status.json"
    status_lock = threading.Lock()
    status = {
        "status": "running",
        "started_at_utc": utc_now(),
        "workers": args.workers,
        "config": str(config),
        "runs": {},
    }
    for stem in args.stems:
        status["runs"][stem] = {
            "run_name": f"{args.run_prefix}{stem}",
            "status": "pending",
            "log": str(log_root / f"{stem}.log"),
        }
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    def run_one(stem: str) -> tuple[str, int, float]:
        run_name = f"{args.run_prefix}{stem}"
        log_path = log_root / f"{stem}.log"
        with status_lock:
            status["runs"][stem].update({
                "status": "running", "started_at_utc": utc_now()})
            status_path.write_text(
                json.dumps(status, indent=2) + "\n", encoding="utf-8")
        started = time.perf_counter()
        command = [
            sys.executable, str(project / "code/pipeline.py"),
            "--run", run_name, "--stem", stem, "--config", str(config),
        ]
        if args.resume_after_m5_prefix:
            command.extend([
                "--resume-after-m5-from",
                f"{args.resume_after_m5_prefix}{stem}",
            ])
        elif args.accepted_only_prefix:
            command.extend([
                "--accepted-only-from",
                f"{args.accepted_only_prefix}{stem}",
            ])
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=project, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False)
        return stem, int(result.returncode), time.perf_counter() - started

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, stem): stem for stem in args.stems}
        for future in as_completed(futures):
            stem, returncode, elapsed = future.result()
            state = "done" if returncode == 0 else "failed"
            status["runs"][stem].update({
                "status": state,
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 3),
                "finished_at_utc": utc_now(),
            })
            if returncode:
                failures.append(stem)
            status_path.write_text(
                json.dumps(status, indent=2) + "\n", encoding="utf-8")
            print(f"{stem}: {state} in {elapsed / 60:.1f} min", flush=True)

    status.update({
        "status": "failed" if failures else "done",
        "failed_stems": failures,
        "finished_at_utc": utc_now(),
    })
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(f"failed recordings: {', '.join(failures)}")


if __name__ == "__main__":
    main()
