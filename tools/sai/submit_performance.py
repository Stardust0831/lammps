#!/usr/bin/env python3
"""Operator-only submission of a frozen, bounded performance experiment."""
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess

from common import CONTROL_FILES, control_digest, digest, read, stamp, write
from endpoint import batch, snapshot


def topology_benchmarks(reference, nodes):
    """Keep the previous experiment's physics and timing, changing only layout."""
    if reference["status"] != "PASS" or reference["plan"]["profile"] != "scaling":
        raise ValueError("topology needs a successful scaling reference")
    if nodes not in (1, 2):
        raise ValueError("topology uses one or two 16-GPU nodes")
    benchmarks = []
    for spec in reference["plan"]["benchmarks"]:
        gpus = spec["nodes"] * spec.get("gpus_per_node", 4)
        if gpus not in (4, 8, 16, 32):
            raise ValueError("unsupported reference layout")
        if gpus <= nodes * 16:
            per_node = min(gpus, 16)
            benchmarks.append(dict(spec, nodes=gpus // per_node, gpus_per_node=per_node))
    layouts = [dict(nodes=1, gpus_per_node=n) for n in (4, 8, 16)]
    if nodes == 2:
        layouts.append(dict(nodes=2, gpus_per_node=16))
    expected = {(kind, series, n * p) for kind in ("snap", "spce") for series in ("strong", "weak")
                for n, p in ((r["nodes"], r["gpus_per_node"]) for r in layouts)}
    actual = [(r["kind"], r["series"], r["nodes"] * r["gpus_per_node"]) for r in benchmarks]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("incomplete reference scaling matrix")
    return layouts, benchmarks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--profile", choices=("pilot", "scaling", "topology"), default="pilot")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--reference", type=Path, help="Completed scaling performance.json to reproduce on full nodes")
    parser.add_argument("--only", nargs="+", help="Run only the named performance points; numerical checks remain enabled")
    parser.add_argument("--first", nargs="+", help="Run the named performance points before the rest")
    parser.add_argument("--ucx-cuda-ipc-cache", choices=("y", "n"), help="Explicit UCX IPC mapping cache setting for this experiment")
    parser.add_argument("--ucx-whole-allocation", choices=("on", "off", "auto"), help="Explicit UCX whole CUDA allocation registration setting")
    parser.add_argument("--disable-cuda-ipc", action="store_true", help="Exclude the CUDA IPC transport for diagnosis")
    parser.add_argument("--qos", choices=("rush-gpu", "huge-gpu"))
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    config = read(source / "site.json")
    allocated_per_node = 16 if args.profile == "topology" else 4
    if args.nodes * allocated_per_node > 16:
        config["qos"] = "huge-gpu"
    if args.qos:
        config["qos"] = args.qos
    directory = args.directory.resolve()
    if not directory.is_relative_to(Path(config["root"]).parent / "lammps-performance"):
        raise ValueError("performance runs must use the project performance directory")
    reference = None
    if args.profile == "topology":
        if args.reference is None:
            raise ValueError("topology needs --reference")
        reference = read(args.reference)
        keys = ("source_sha", "image", "image_sha256", "recipe_sha256", "fixed_module", "module_root",
                "runtime", "catalog_sha256", "build_evidence", "build_evidence_sha256")
        selected = {key: reference["plan"][key] for key in keys}
        selected["control_digest"] = control_digest(source)
        if reference["plan"]["partition"] != config["partition"]:
            raise ValueError("reference used a different partition")
        for name, expected in reference["plan"]["control_files"].items():
            if name.startswith("benchmarks/") and digest(source / name) != expected:
                raise ValueError("benchmark input differs from reference: " + name)
    else:
        if args.reference is not None:
            raise ValueError("--reference is only supported by topology")
        selected = snapshot(config)
    layouts = [n for n in (1, 2, 4, 8) if n <= args.nodes]
    benchmarks = []
    if args.profile == "topology":
        layouts, benchmarks = topology_benchmarks(reference, args.nodes)
    elif args.profile == "pilot":
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
    if args.only:
        if not set(args.only).issubset({r["name"] for r in benchmarks}):
            raise ValueError("unknown requested performance point")
        benchmarks = [r for r in benchmarks if r["name"] in args.only]
    if args.first:
        if not set(args.first).issubset({r["name"] for r in benchmarks}):
            raise ValueError("unknown priority performance point")
        priority = {name: index for index, name in enumerate(args.first)}
        benchmarks.sort(key=lambda row: priority.get(row["name"], len(priority)))
    runtime_environment = {}
    if args.ucx_cuda_ipc_cache is not None:
        runtime_environment["UCX_CUDA_IPC_CACHE"] = args.ucx_cuda_ipc_cache
    if args.ucx_whole_allocation is not None:
        runtime_environment["UCX_CUDA_COPY_REG_WHOLE_ALLOC"] = args.ucx_whole_allocation
    if args.disable_cuda_ipc:
        runtime_environment["UCX_TLS"] = "^cuda_ipc"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "scratch").mkdir(mode=0o700)
    (directory / "control").mkdir()
    for name in (*CONTROL_FILES, "performance.py", "submit_performance.py"):
        shutil.copyfile(source / name, directory / "control" / name)
    shutil.copytree(source / "benchmarks", directory / "control/benchmarks")
    if reference is not None:
        shutil.copyfile(args.reference, directory / "reference-performance.json")
    hashes = {str(p.relative_to(directory / "control")): digest(p)
              for p in sorted((directory / "control").rglob("*")) if p.is_file()}
    plan = dict(selected, created_at=stamp(), profile=args.profile, nodes=args.nodes,
                allocation_gpus_per_node=allocated_per_node,
                runtime_environment=runtime_environment,
                layouts=layouts, benchmarks=benchmarks, partition=config["partition"],
                account=config["account"], qos=config["qos"], control_files=hashes,
                numerical_scope="10-step CPU/Kokkos force and thermo comparisons; large NVT finite-state checks",
                calibration=str(args.calibration) if args.calibration else None,
                run_id=directory.name, action_url="https://github.com/Stardust0831/lammps",
                report_url="https://github.com/Stardust0831/lammps/tree/feat/sai-multigpu-tester/tools/sai")
    if reference is not None:
        plan.update(reference_sha256=digest(directory / "reference-performance.json"),
                    reference_run=reference["plan"]["run_id"], gpu_assignment="cpu-local")
    write(directory / "plan.json", plan)
    job = batch(plan, dict(config, minutes=45), directory)
    job = job.replace("sai-lammps-test", "sai-lmp-" + args.profile)
    job = job.replace("#SBATCH --nodes=2", f"#SBATCH --nodes={args.nodes}")
    job = job.replace("#SBATCH --ntasks=8\n", f"#SBATCH --ntasks={args.nodes * allocated_per_node}\n")
    job = job.replace("#SBATCH --ntasks-per-node=4", f"#SBATCH --ntasks-per-node={allocated_per_node}")
    job = job.replace("#SBATCH --gres=gpu:4", f"#SBATCH --gres=gpu:{allocated_per_node}")
    job = job.replace("control/suite.py", "control/performance.py")
    if runtime_environment:
        exports = "\n".join("export " + key + "=" + shlex.quote(value) for key, value in runtime_environment.items())
        job = job.replace("exec /usr/bin/python3", exports + "\nexec /usr/bin/python3")
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
