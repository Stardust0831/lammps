#!/usr/bin/env python3
"""Operator-only submission of a frozen, bounded performance experiment."""
import argparse
from pathlib import Path
import shutil
import subprocess

from common import CONTROL_FILES, digest, read, stamp, write
from endpoint import batch, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nodes", type=int, choices=(2, 4, 8), default=2)
    parser.add_argument("--profile", choices=("pilot", "scaling"), default="pilot")
    parser.add_argument("--calibration", type=Path)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config = read(source / "site.json")
    if args.nodes * 4 > 16:
        config["qos"] = "huge-gpu"
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path(config["root"]).parent / "lammps-performance"):
        raise ValueError("performance runs must use the project performance directory")
    selected = snapshot(config)
    layouts = [n for n in (1, 2, 4, 8) if n <= args.nodes]
    benchmarks = []
    if args.profile == "pilot":
        for kind, label, dims in (("snap", "snap-base", (64, 64, 64)),
                                  ("spce", "spce-base", (3, 3, 3)),
                                  ("spce", "spce-large", (6, 6, 6))):
            for nodes in layouts:
                benchmarks.append(dict(name=f"{label}-{nodes * 4}gpu", kind=kind, dims=dims,
                                       nodes=nodes, steps=100, warmup=100, repeats=2, series="pilot"))
    else:
        if args.calibration is None:
            raise ValueError("scaling needs a successful pilot")
        pilot = read(args.calibration)
        if pilot["status"] != "PASS" or pilot["plan"]["image_sha256"] != selected["image_sha256"]:
            raise ValueError("pilot failed or used a different image")
        for kind, base, strong in (("snap", (64, 64, 64), (128, 128, 64)),
                                   ("spce", (3, 3, 3), (6, 6, 6))):
            sample = next(r for r in pilot["measurements"] if r["name"] == kind + "-base-4gpu")
            # At least ~30 s per base-size sample; fixed steps across each scaling series.
            steps = max(100, min(5000, int(30 / sample["median_seconds"] * 100 / 50 + 1) * 50))
            for nodes in layouts:
                factors = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1), 8: (2, 2, 2)}[nodes]
                for series, dims in (("strong", strong), ("weak", tuple(a * b for a, b in zip(base, factors)))):
                    benchmarks.append(dict(name=f"{kind}-{series}-{nodes * 4}gpu", kind=kind,
                                           dims=dims, nodes=nodes, steps=steps if series == "weak" else max(100, steps // 4),
                                           warmup=100, repeats=3, series=series))
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "scratch").mkdir(mode=0o700)
    (directory / "control").mkdir()
    for name in (*CONTROL_FILES, "performance.py", "submit_performance.py"):
        shutil.copyfile(source / name, directory / "control" / name)
    shutil.copytree(source / "benchmarks", directory / "control/benchmarks")
    hashes = {str(p.relative_to(directory / "control")): digest(p)
              for p in sorted((directory / "control").rglob("*")) if p.is_file()}
    plan = dict(selected, created_at=stamp(), profile=args.profile, nodes=args.nodes,
                layouts=layouts, benchmarks=benchmarks, partition=config["partition"],
                account=config["account"], qos=config["qos"], control_files=hashes,
                numerical_scope="10-step CPU/Kokkos force and thermo comparisons; large NVT finite-state checks",
                calibration=str(args.calibration) if args.calibration else None,
                run_id=directory.name, action_url="https://github.com/Stardust0831/lammps",
                report_url="https://github.com/Stardust0831/lammps/tree/feat/sai-multigpu-tester/tools/sai")
    write(directory / "plan.json", plan)
    job = batch(plan, dict(config, minutes=45), directory)
    job = job.replace("sai-lammps-test", "sai-lmp-" + args.profile)
    job = job.replace("#SBATCH --nodes=2", f"#SBATCH --nodes={args.nodes}")
    job = job.replace("#SBATCH --ntasks=8\n", f"#SBATCH --ntasks={args.nodes * 4}\n")
    job = job.replace("control/suite.py", "control/performance.py")
    path = directory / "job.sbatch"
    path.write_text(job)
    subprocess.run(["bash", "-n", str(path)], check=True)
    submitted = subprocess.run(["sbatch", "--hold", "--parsable", str(path)], cwd=directory,
                               capture_output=True, text=True)
    write(directory / "submission.json", dict(returncode=submitted.returncode,
                                               stdout=submitted.stdout, stderr=submitted.stderr))
    if submitted.returncode:
        raise RuntimeError(submitted.stderr)
    jobid = submitted.stdout.strip().split(";")[0]
    if not jobid.isdigit():
        raise ValueError("unexpected Slurm submission receipt")
    plan["job"] = jobid
    write(directory / "plan.json", plan)
    subprocess.run(["scontrol", "release", jobid], check=True)
    print("Submitted", jobid, directory, flush=True)


if __name__ == "__main__":
    main()
