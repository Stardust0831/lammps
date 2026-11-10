#!/usr/bin/env python3
"""Validate completed scaling evidence and write plots and dashboard JSON."""
import argparse
import csv
import math
from pathlib import Path
import re
import statistics

from common import digest, read, write


def build_report(run, accounting, output):
    result = read(run / "performance.json")
    plan = result["plan"]
    if result["status"] != "PASS":
        raise ValueError("the run did not pass")
    if (accounting["State"] != "COMPLETED" or accounting["ExitCode"] != "0:0"
            or accounting["JobIDRaw"] != plan["job"] or accounting["Account"] != plan["account"]
            or not accounting["WorkDir"].endswith("/" + plan["run_id"])):
        raise ValueError("Slurm accounting does not confirm the expected successful run")
    for name, expected in plan["control_files"].items():
        if digest(run / "control" / name) != expected:
            raise ValueError("changed frozen controller or input: " + name)
    gates = result["numerical_checks"]
    if len(gates) != 2 * len(plan["layouts"]) or any(g["status"] != "PASS" for g in gates):
        raise ValueError("incomplete numerical checks")
    records = result["measurements"]
    if [r["name"] for r in records] != [r["name"] for r in plan["benchmarks"]]:
        raise ValueError("missing or reordered measurements")
    for row in records:
        if len(row["loop_seconds"]) != row["repeats"] or not all(math.isfinite(t) and t > 0 for t in row["loop_seconds"]):
            raise ValueError("invalid measured samples")
    energy_checks = {}
    for kind in ("snap", "spce"):
        energies = [r["states"]["initial"]["values"][0] for r in records
                    if r["kind"] == kind and r["series"] == "strong"]
        spread = max(energies) - min(energies)
        if not math.isfinite(spread) or spread > 1e-6:
            raise ValueError("initial strong-scaling energy differs between layouts")
        energy_checks[kind] = dict(max_minus_min=spread, tolerance=1e-6,
                                   unit="eV/atom" if kind == "snap" else "kcal/mol/atom")
    output.mkdir(parents=True, exist_ok=True)
    screen = (run / "cases/snap-cpu-reference/screen.txt").read_text()
    banner = re.search(r"^LAMMPS \((.+)\)$", screen, re.M)
    if not banner or plan["source_sha"][:8] not in banner.group(1):
        raise ValueError("the executable banner does not match the recorded source")
    summary = dict(date=result["finished_at"], commit=plan["source_sha"], branch="develop",
                   version=banner.group(1),
                   compiler="See accepted image build provenance", operating_system=result["hardware"][0]["operating_system"],
                   cpu=result["hardware"][0]["cpu"], gpus="V100-SXM2-16GB; four per node", mpi=result["mpi"],
                   url="https://github.com/Stardust0831/lammps/tree/sai-multigpu-reports/performance/"
                   + result["finished_at"][:10].replace("-", ""),
                   image_sha256=plan["image_sha256"], binary_modified="-modified" in banner.group(1),
                   reporter_sha256=digest(Path(__file__)),
                   sample_policy="Three sequential segments after warmup; SAI plots use medians and min/max ranges. "
                   "The upstream dashboard currently selects the best sample.", results={})
    table = []
    for row in records:
        median = statistics.median(row["loop_seconds"])
        baseline = next(r for r in records if r["kind"] == row["kind"] and r["series"] == row["series"] and r["nodes"] == 1)
        base_time = statistics.median(baseline["loop_seconds"])
        speedup = base_time / median
        efficiency = speedup / row["nodes"] if row["series"] == "strong" else speedup
        table.append(dict(case=row["name"], series=row["series"], atoms=row["natoms"],
                          gpus=row["gpus"], steps=row["steps"], median_seconds=median,
                          min_seconds=min(row["loop_seconds"]), max_seconds=max(row["loop_seconds"]),
                          ns_per_day=86400 / median * row["steps"] * (0.5e-6 if row["kind"] == "snap" else 2e-6),
                          relative_speedup=speedup, efficiency=efficiency,
                          full_process_seconds=row["wall_seconds"]))
        summary["results"][row["name"]] = dict(input=row["kind"], size=str(row["natoms"]),
              precision="double", package="kokkos", api="cuda", category="kokkos-cuda",
              flags="newton on neigh half comm device gpu/aware on", ranks=row["gpus"],
              threads=1, gpus=row["gpus"], natoms=row["natoms"], steps=row["steps"],
              samples=row["atom_steps_per_second"], loop_time=row["loop_seconds"], status="ok")
    write(output / "performance-summary.json", summary)
    write(output / "performance.json", result)
    write(output / "accounting.json", accounting)
    write(output / "large-system-checks.json", energy_checks)
    write(output / "table.json", table)
    with (output / "table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for rowid, kind in enumerate(("snap", "spce")):
        for colid, series in enumerate(("strong", "weak")):
            selected = [r for r in table if r["case"].startswith(kind + "-") and r["series"] == series]
            ax = axes[rowid, colid]
            x = [r["gpus"] for r in selected]
            base = selected[0]["median_seconds"]
            if series == "strong":
                y = [r["relative_speedup"] for r in selected]
                lo = [r["relative_speedup"] - base / r["max_seconds"] for r in selected]
                hi = [base / r["min_seconds"] - r["relative_speedup"] for r in selected]
                ax.plot(x, [n / x[0] for n in x], "--", color="gray", label="Ideal")
                ax.set_ylabel("Speedup relative to 4 GPUs")
            else:
                y = [r["efficiency"] * 100 for r in selected]
                lo = [r["efficiency"] * 100 - base / r["max_seconds"] * 100 for r in selected]
                hi = [base / r["min_seconds"] * 100 - r["efficiency"] * 100 for r in selected]
                ax.axhline(100, ls="--", color="gray", label="Ideal")
                ax.set_ylabel("Weak-scaling efficiency (%)")
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=4, label="Median; min/max samples")
            ax.set_xticks(x)
            ax.set_xlabel("V100 GPUs (4 per node)")
            ax.set_title(("Tungsten SNAP" if kind == "snap" else "SPC/E water + PPPM") + " - " + series)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("SAI LAMMPS Kokkos CUDA scaling - " + plan["source_sha"][:12])
    fig.savefig(output / "scaling.png", dpi=180)
    fig.savefig(output / "scaling.svg")
    plt.close(fig)
    gpu_hours = float(accounting["ElapsedRaw"]) * plan["nodes"] * 4 / 3600
    lines = ["# SAI LAMMPS multi-node performance trial", "",
             "![Scaling curves](scaling.png)", "",
             "Source: `" + plan["source_sha"] + "`. Accepted site image with existing DeePMD integration; "
             "the binary reports `-modified`. No LAMMPS source changes were made for this experiment.", "",
             "Slurm job `" + plan["job"] + "`: `" + accounting["State"] + "`, exit `" + accounting["ExitCode"]
             + "`, " + accounting["Elapsed"] + ", " + f"{gpu_hours:.2f} allocated GPU-hours. "
             "This includes reserved GPUs while smaller layouts run.", "",
             "One MPI rank per GPU, four GPUs per node, one OpenMP thread per rank. "
             "MPI binds each rank to four physical cores, following the site's one-rank-per-GPU mapping. "
             "Three consecutive timed segments follow 100 warmup steps. Times exclude setup; "
             "full process wall times remain in the CSV and raw evidence. Nodes were not reserved exclusively.", "",
             "Kokkos CUDA runs asynchronously. The raw Pair/Comm/etc. sections can charge GPU waits "
             "to a later section and must not be read as network-only or kernel-only costs. "
             "The LAMMPS Kokkos guide recommends CUDA_LAUNCH_BLOCKING=1 for accurate section timings; "
             "that diagnostic mode was not used here because it changes execution performance. "
             "The curves use complete loop times.", "",
             "| Case | Atoms | GPUs | Steps/sample | Median s | Min-max s | Strong speedup / weak efficiency |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in table:
        metric = f"{row['relative_speedup']:.2f}x" if row["series"] == "strong" else f"{row['efficiency'] * 100:.1f}%"
        lines.append(f"| {row['case']} | {row['atoms']} | {row['gpus']} | {row['steps']} | "
                     f"{row['median_seconds']:.3f} | {row['min_seconds']:.3f}-{row['max_seconds']:.3f} | {metric} |")
    lines += ["", f"All {len(gates)} small-system CPU/GPU numerical checks passed. "
              "These compare initial and 10-step energy, pressure tensor and every atomic force. "
              "Large-system runs check atom counts, finite states, temperature bounds and "
              "initial potential energy across strong-scaling layouts (absolute tolerance 1e-6 per atom); "
              "they are performance trials, not converged production trajectories.", "",
              "[Dashboard JSON](performance-summary.json) | [Full measurements](performance.json) | "
              "[CSV](table.csv) | [Slurm accounting](accounting.json) | "
              "[Raw evidence](raw.tar.gz) | [Hashes](manifest.json)", "",
              "The existing upstream performance importer accepts the JSON. Its current display "
              "uses the best sample; this report uses medians and shows every sample range. "
              "These are adapted W SNAP and standard SPC/E inputs, not formal UK procurement results.", ""]
    (output / "README.md").write_text("\n".join(lines))
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_report(args.run, read(args.accounting), args.output)
