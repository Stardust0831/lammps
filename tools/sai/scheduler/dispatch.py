#!/usr/bin/env python3
"""Request the existing SAI regression workflow from a local timer."""
import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import sys
import urllib.error
import urllib.request


WORKFLOW = "/repos/Stardust0831/lammps/actions/workflows/sai-multigpu.yml"
VISIBILITY_GRACE = timedelta(minutes=15)


class GitHub:
    def __init__(self, token_file):
        info = token_file.lstat()
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or info.st_mode & 0o077):
            raise ValueError("token must be an owner-only regular file owned by this account")
        self.token = token_file.read_text().strip()
        if not self.token or any(char.isspace() for char in self.token):
            raise ValueError("empty or invalid dispatch token")

    def request(self, path, data=None):
        request = urllib.request.Request(
            "https://api.github.com" + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Authorization": "Bearer " + self.token,
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "User-Agent": "lammps-sai-local-scheduler",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read()
        return json.loads(content) if content else None

    def runs(self):
        workflow = self.request(WORKFLOW)
        if workflow["state"] != "active":
            raise ValueError("SAI regression workflow is not active")
        # Query only unfinished runs, including runs awaiting approval. A long
        # queue must remain visible even after many newer completed runs.
        rows = []
        for status in ("queued", "in_progress", "waiting", "requested", "pending"):
            data = self.request(WORKFLOW + "/runs?per_page=1&status=" + status)
            rows.extend(data["workflow_runs"])
        return rows

    def dispatch(self):
        self.request(WORKFLOW + "/dispatches",
                     {"ref": "develop", "inputs": {"force": "false"}})


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def tick(root, github, now):
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (root / "dispatch.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("SCHEDULE locked: another local check is running", flush=True)
            return "locked"
        path = root / "receipt.json"
        receipt = json.loads(path.read_text()) if path.exists() else None
        if receipt:
            requested = datetime.fromisoformat(receipt["requested_at"])
            if now - requested < VISIBILITY_GRACE:
                print("SCHEDULE awaiting-dispatch-visibility", flush=True)
                return "awaiting-dispatch-visibility"
        if any(run["status"] != "completed" for run in github.runs()):
            print("SCHEDULE busy: an Actions run is already active", flush=True)
            return "busy"
        # Save before POST: if its response is lost, wait before retrying.
        # The workflow and cluster endpoint also serialize and reuse GPU work.
        record = {"requested_at": now.isoformat(), "accepted": False}
        save(path, record)
        github.dispatch()
        record["accepted"] = True
        save(path, record)
        print("DISPATCH_ACCEPTED develop force=false", flush=True)
        return "dispatched"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, default=Path(os.environ.get(
        "LAMMPS_SAI_TOKEN_FILE", Path.home() / ".config/lammps-sai-scheduler/token")))
    parser.add_argument("--state-root", type=Path,
                        default=Path.home() / ".local/state/lammps-sai-scheduler")
    parser.add_argument("--check", action="store_true", help="check API access without dispatching")
    args = parser.parse_args()
    os.umask(0o077)
    github = GitHub(args.token_file)
    if args.check:
        runs = github.runs()
        print(f"API access verified; {len(runs)} active run(s). No request sent.")
    else:
        tick(args.state_root, github, datetime.now(timezone.utc))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as error:
        print(f"GitHub HTTP {error.code}; check Actions permission and token expiry.", file=sys.stderr)
        sys.exit(1)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Local scheduler failed: {error}", file=sys.stderr)
        sys.exit(1)
