#!/usr/bin/env python3
"""Restricted SSH entry point; only the installed operator-owned suite is run."""
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

from common import CONTROL_FILES, control_digest, digest, read, reports, stamp, write

CONTROL = Path(__file__).resolve().parent


def checked(value, expression, description):
    if not re.fullmatch(expression, value):
        raise ValueError("invalid " + description)
    return value


def snapshot(config):
    module_root = Path(config["module_root"])
    module = (module_root / "lammps/develop-beta-later").resolve(strict=True)
    if module.parent != module_root / "lammps":
        raise ValueError("module escaped its fixed directory")
    generation = re.search(r'^set sai_beta_generation "([A-Za-z0-9_.-]+)"$', module.read_text(), re.M)
    if not generation:
        raise ValueError("unrecognized beta module")
    runtime = Path(config["beta_root"]) / "runtimes" / generation.group(1)
    catalog = read(runtime / "catalog.json")
    row = catalog["applications"]["lammps"]["partitions"][config["partition"]]
    if not row["build_verified"] or not row["scientific_verified"]:
        raise ValueError("LAMMPS image has not passed the build workflow")
    identity = row["identity"]
    if identity["source_ref"] != "develop" or identity["repository"] != "lammps/lammps":
        raise ValueError("not an upstream develop image")
    return dict(source_sha=identity["source_sha"], image=row["image"],
                image_sha256=row["sha256"], recipe_sha256=identity["recipe_sha256"],
                fixed_module="lammps/" + module.name, module_root=str(module_root),
                runtime=str(runtime), catalog_sha256=digest(runtime / "catalog.json"),
                build_evidence=row["evidence"], build_evidence_sha256=row["evidence_sha256"],
                control_digest=control_digest(CONTROL))


def batch(plan, config, directory):
    quote = shlex.quote
    return f'''#!/bin/bash
#SBATCH --job-name=sai-lammps-test
#SBATCH --partition={config['partition']}
#SBATCH --qos={config['qos']}
#SBATCH --account={config['account']}
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:{config['minutes']:02d}:00
#SBATCH --chdir={directory}
#SBATCH --output={directory}/slurm-%j.log
#SBATCH --export=NIL
set -eo pipefail
export HOME={quote(str(Path.home()))}
export USER=${{SLURM_JOB_USER:?}} LOGNAME=${{SLURM_JOB_USER:?}}
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH= LD_PRELOAD= PYTHONPATH=
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export TMPDIR={quote(str(directory / 'scratch'))}
export TMP="$TMPDIR" TEMP="$TMPDIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
export OMPI_MCA_plm_slurm_args=--external-launcher PRTE_MCA_plm_slurm_args=--external-launcher
unset PYTHONHOME APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
source /etc/profile.d/lmod.sh
module purge
module use {quote(plan['module_root'])}
module load {quote(plan['fixed_module'])}
module -t list > modules.txt 2>&1
scontrol show job -d "$SLURM_JOB_ID" > allocation.txt
exec /usr/bin/python3 -B {quote(str(directory / 'control/suite.py'))} run {quote(str(directory))}
'''


def accounting(job):
    output = subprocess.check_output(["sacct", "-X", "-n", "-P", "-j", job,
        "--format=JobIDRaw,State,ExitCode,Partition,Account,WorkDir,Elapsed,NodeList,AllocTRES,TotalCPU"], text=True)
    matches = [line.split("|") for line in output.splitlines() if line.split("|")[0] == job]
    return matches[0] if matches else None


def status(directory, config):
    plan = read(directory / "plan.json")
    final = directory / "final.json"
    if final.exists():
        return read(final)
    job = plan["job"]
    row = accounting(job)
    if not row or row[1].split("+")[0] in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED"}:
        return dict(run_id=plan["run_id"], job=job, terminal=False,
                    state=row[1] if row else "ACCOUNTING_PENDING", source_sha=plan["source_sha"])
    write(directory / "accounting.json", row)
    expected = ["COMPLETED", "0:0", config["partition"], config["account"], str(directory)]
    summary_path = directory / "results/summary.json"
    cases_path = directory / "results/cases.json"
    if not summary_path.exists():
        result = reports(directory, [dict(name="infrastructure", status="error", seconds=0,
                         message="Slurm ended before producing results: " + " | ".join(row))])
    else:
        result = read(summary_path)
    if row[1:6] != expected and result["status"] == "PASS":
        cases = read(cases_path)
        cases.append(dict(name="slurm-accounting", status="error", seconds=0,
                          message="unexpected Slurm state or allocation: " + " | ".join(row)))
        result = reports(directory, cases, result["machine"], result["version"])
    final_record = dict(run_id=plan["run_id"], job=job, terminal=True,
                        state=row[1], source_sha=plan["source_sha"], status=result["status"],
                        action_url=plan["action_url"], completed_at=result["date"],
                        tests=result["tests"], failures=result["failures"], errors=result["errors"])
    # Archive includes the exact inputs, controls, outputs, GPU mappings and accounting.
    raw = directory / "results/raw.tar.gz"
    excluded = {"results/raw.tar.gz", "results/manifest.json", "final.json"}
    files = {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob("*"))
             if p.is_file() and not p.is_symlink() and "scratch" not in p.relative_to(directory).parts
             and str(p.relative_to(directory)) not in excluded}
    with tarfile.open(raw, "w:gz") as archive:
        for name in files:
            archive.add(directory / name, arcname=name, recursive=False)
    write(directory / "results/manifest.json", dict(files=files, raw_sha256=digest(raw),
          raw_bytes=raw.stat().st_size, source_sha=plan["source_sha"], slurm=row))
    write(final, final_record)
    return final_record


def start(config, run_id, source_sha, controller_sha, force):
    checked(run_id, r"github-[0-9]+-[0-9]+|pilot-[A-Za-z0-9-]+", "run ID")
    checked(source_sha, r"[0-9a-f]{40}", "source SHA")
    checked(controller_sha, r"[0-9a-f]{40}", "controller SHA")
    if force not in ("once", "force"):
        raise ValueError("invalid retest policy")
    root = Path(config["root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state_path = root / "state.json"
        state = read(state_path) if state_path.exists() else {}
        if state.get("active"):
            active = status(root / "runs" / state["active"], config)
            # Keep completed work resumable until the public report is acknowledged.
            return dict(active, resumed=True)
        if force == "once" and state.get("last", {}).get("source_sha") == source_sha:
            return dict(state["last"], skipped=True)
        selected = snapshot(config)
        if selected["source_sha"] != source_sha:
            return dict(pending=True, expected_source=source_sha, available_source=selected["source_sha"])
        directory = root / "runs" / run_id
        if directory.exists():
            if (directory / "plan.json").exists() and read(directory / "plan.json").get("job"):
                return status(directory, config)
            raise RuntimeError("run directory exists without a durable job handle; inspect before retry")
        directory.mkdir(parents=True)
        (directory / "scratch").mkdir(mode=0o700)
        (directory / "control").mkdir()
        for name in CONTROL_FILES:
            shutil.copyfile(CONTROL / name, directory / "control" / name)
        action = ("https://github.com/" + config["repository"] + "/actions/runs/" + run_id.split("-")[1]
                  if run_id.startswith("github-") else "https://github.com/" + config["repository"])
        plan = dict(selected, run_id=run_id, controller_sha=controller_sha,
                    action_url=action, report_url=config["report_url"], partition=config["partition"],
                    created_at=stamp())
        write(directory / "plan.json", plan)
        script = directory / "job.sbatch"
        script.write_text(batch(plan, config, directory))
        subprocess.run(["bash", "-n", str(script)], check=True)
        # Hold until the job number is durable; a restarted relay resumes it.
        submission = subprocess.run(["sbatch", "--hold", "--parsable", str(script)],
                                    cwd=directory, text=True, capture_output=True)
        write(directory / "submission.json", dict(returncode=submission.returncode,
              stdout=submission.stdout, stderr=submission.stderr))
        if submission.returncode:
            raise RuntimeError("Slurm submission failed: " + submission.stderr)
        submitted = submission.stdout.strip().split(";")[0]
        checked(submitted, r"[0-9]+", "Slurm job ID")
        plan["job"] = submitted
        write(directory / "plan.json", plan)
        write(state_path, dict(state, active=run_id))
        subprocess.run(["scontrol", "release", submitted], check=True)
        return dict(run_id=run_id, job=submitted, terminal=False, state="SUBMITTED", source_sha=source_sha)


def main(arguments):
    config = read(CONTROL / "site.json")
    if arguments == ["inspect"]:
        return snapshot(config)
    if len(arguments) == 5 and arguments[0] == "start":
        return start(config, *arguments[1:])
    if len(arguments) == 2 and arguments[0] in ("status", "fetch", "ack"):
        run_id = checked(arguments[1], r"github-[0-9]+-[0-9]+|pilot-[A-Za-z0-9-]+", "run ID")
        directory = Path(config["root"]) / "runs" / run_id
        with (Path(config["root"]) / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = status(directory, config)
            if arguments[0] == "ack":
                if not result["terminal"]:
                    raise ValueError("cannot acknowledge an unfinished run")
                state_path = Path(config["root"]) / "state.json"
                state = read(state_path)
                if state.get("active") == run_id:
                    state["last"] = result
                    state.pop("active")
                    write(state_path, state)
                elif state.get("last", {}).get("run_id") != run_id:
                    raise ValueError("acknowledgement does not match active or last run")
                return dict(acknowledged=True, run_id=run_id)
        if arguments[0] == "status":
            return result
        if not result["terminal"]:
            raise ValueError("run has not finished")
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
            for name in ("junit.xml", "summary.json", "cases.json", "index.html", "raw.tar.gz", "manifest.json"):
                archive.add(directory / "results" / name, arcname=name, recursive=False)
        return None
    raise ValueError("unsupported tester command")


if __name__ == "__main__":
    try:
        args = shlex.split(os.environ["SSH_ORIGINAL_COMMAND"]) if "SSH_ORIGINAL_COMMAND" in os.environ else sys.argv[1:]
        response = main(args)
        if response is not None:
            print(json.dumps(response))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from None
