#!/usr/bin/env python3
"""Test the accepted daily build and resume the fixed SAI regression suite."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from common import control_digest, read


def output(**values):
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
            for key, value in values.items():
                stream.write(f"{key}={value}\n")


def remote(*arguments, binary=False):
    command = ["ssh", "-F", os.environ["SAI_SSH_CONFIG"], "sai-lammps-tester", " ".join(arguments)]
    last_error = None
    for attempt in range(3):
        try:
            result = subprocess.run(command, capture_output=True, timeout=180)
            if result.returncode:
                if result.returncode != 255:
                    raise ValueError(result.stderr.decode(errors="replace"))
                raise RuntimeError(result.stderr.decode(errors="replace"))
            return result.stdout if binary else json.loads(result.stdout)
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(5)
    raise RuntimeError(str(last_error))


def start_accepted(run_id, controller_sha, policy):
    for _ in range(3):
        selected = remote("inspect")
        if selected["control_digest"] != control_digest(Path(__file__).resolve().parent):
            raise ValueError("operator-installed controller differs from this workflow; deploy it first")
        print("Accepted develop build:", selected["source_sha"],
              "Image SHA256:", selected["image_sha256"], flush=True)
        state = remote("start", run_id, selected["source_sha"], controller_sha, policy)
        if not state.get("pending"):
            return state
        print("Accepted build changed during selection; reading the new build.", flush=True)
    raise RuntimeError("accepted build changed repeatedly during selection; retry this workflow")


def collect(raw, destination, expected_sha):
    destination = Path(destination)
    destination.mkdir(exist_ok=True)
    names = {"junit.xml", "summary.json", "cases.json", "index.html", "raw.tar.gz", "manifest.json"}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) != len(names) or {m.name for m in members} != names:
            raise ValueError("unexpected report members")
        for member in members:
            if not member.isfile() or member.size > 50_000_000:
                raise ValueError("unsafe or oversized report member")
            (destination / member.name).write_bytes(archive.extractfile(member).read())
    summary = read(destination / "summary.json")
    if summary["commit"] != expected_sha:
        raise ValueError("downloaded source commit differs from completed run")
    manifest = read(destination / "manifest.json")
    if hashlib.sha256((destination / "raw.tar.gz").read_bytes()).hexdigest() != manifest["raw_sha256"]:
        raise ValueError("raw archive checksum mismatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ack", action="store_true", help="acknowledge a published report")
    args = parser.parse_args()
    if args.ack:
        remote("ack", read(Path("artifacts") / "summary.json")["run_id"])
        return 0
    run_id = "github-" + os.environ["GITHUB_RUN_ID"] + "-" + os.environ["GITHUB_RUN_ATTEMPT"]
    state = start_accepted(run_id, os.environ["GITHUB_SHA"], "force" if args.force else "once")
    if state.get("skipped"):
        message = ("Accepted image and test configuration unchanged; no GPU job submitted. Previous result: "
                   + state["status"] + " " + state["action_url"])
        print(message, flush=True)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(message + "\n")
        output(publish="false", skipped="true")
        return 0
    deadline = time.monotonic() + 70 * 60
    previous = None
    while not state["terminal"]:
        message = "Slurm " + state["job"] + ": " + state["state"]
        if previous != message:
            print(message, flush=True)
            previous = message
        if time.monotonic() > deadline:
            raise TimeoutError("job is still active; the next poll will resume it")
        time.sleep(15)
        state = remote("status", state["run_id"])
    result = collect(remote("fetch", state["run_id"], binary=True), "artifacts", state["source_sha"])
    print(json.dumps(result, indent=2), flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(
            f"SAI multi-GPU tests: **{result['status']}**\n\n"
            f"Accepted develop build `{result['commit']}`; {result['passed']}/{result['tests']} passed, "
            f"{result['failures']} failures, {result['errors']} errors.\n\n"
            f"Slurm job: {result['slurm_job']}. [Public report]({result['url']})\n")
    output(publish="true", skipped="false")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
