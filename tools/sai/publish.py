#!/usr/bin/env python3
"""Publish a completed report atomically as one Git commit."""
import base64
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from common import read


def main():
    source = Path("artifacts").resolve()
    summary = read(source / "summary.json")
    config = read(Path(__file__).with_name("site.json"))
    repository, branch = config["repository"], config["report_branch"]
    if os.environ["GITHUB_REPOSITORY"] != repository:
        raise ValueError("publication is restricted to the configured fork")
    token = base64.b64encode(("x-access-token:" + os.environ["GH_TOKEN"]).encode()).decode()
    env = dict(os.environ, GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraheader",
               GIT_CONFIG_VALUE_0="AUTHORIZATION: basic " + token, GIT_TERMINAL_PROMPT="0")
    with tempfile.TemporaryDirectory(prefix="lammps-reports-", dir=os.environ["RUNNER_TEMP"]) as tmp:
        def git(*arguments, allowed=False):
            return subprocess.run(["git", *arguments], cwd=tmp, env=env, check=not allowed,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        git("init", "-q")
        git("remote", "add", "origin", "https://github.com/" + repository + ".git")
        found = git("ls-remote", "--heads", "origin", branch).stdout.strip()
        if found:
            git("fetch", "-q", "--depth=1", "origin", branch)
            git("checkout", "-q", "-B", branch, "FETCH_HEAD")
            previous = Path(tmp) / "summary.json"
            if previous.exists() and read(previous)["date"] > summary["date"]:
                raise ValueError("refusing to replace a newer completed report")
        else:
            git("checkout", "-q", "--orphan", branch)
        for name in ("summary.json", "junit.xml", "cases.json", "index.html", "raw.tar.gz", "manifest.json"):
            shutil.copyfile(source / name, Path(tmp) / name)
        (Path(tmp) / ".nojekyll").write_text("")
        (Path(tmp) / "README.md").write_text(
            "# SAI LAMMPS multi-GPU tests\n\n"
            "[Readable report](" + config["report_url"] + ") | [Summary](summary.json) | "
            "[JUnit](junit.xml) | [Raw evidence](raw.tar.gz) | [Hashes](manifest.json)\n\n"
            "LJ and PPPM comparisons using the GPU package and Kokkos CUDA on one and two nodes. "
            "The source commit identifies the upstream develop code in the tested accepted image. "
            "Unchanged upstream commits are skipped unless a retest is requested. "
            "CPU execution supplies numerical controls; timing is diagnostic, not a scaling claim.\n\n"
            "[Workflow and operator documentation](https://github.com/" + repository + "/tree/develop/tools/sai)\n")
        git("config", "user.name", "github-actions[bot]")
        git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
        git("add", ".")
        changed = git("diff", "--cached", "--quiet", allowed=True)
        if changed.returncode:
            git("commit", "-q", "-m", "SAI GPU regression " + summary["commit"][:12] + " " + summary["status"])
        pushed = git("push", "origin", "HEAD:" + branch, allowed=True)
        if pushed.returncode:
            # Do not echo a credential-bearing Git configuration on error.
            raise RuntimeError("report push failed: " + pushed.stderr)
        print("Published report:", config["report_url"], flush=True)


if __name__ == "__main__":
    main()
