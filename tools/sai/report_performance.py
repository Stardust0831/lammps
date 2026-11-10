#!/usr/bin/env python3
"""Validate completed scaling evidence and write plots and dashboard JSON."""
import argparse
import csv
import math
from pathlib import Path
import re
import statistics

from common import digest, read, write


def build_report(run, accounting, output, build_summary=None):
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
        if row["gpus"] != row["nodes"] * row.get("gpus_per_node", 4):
            raise ValueError("GPU count does not match node layout")
    if "reference_sha256" in plan:
        if digest(run / "reference-performance.json") != plan["reference_sha256"]:
            raise ValueError("reference experiment changed")
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
                   cpu=result["hardware"][0]["cpu"],
                   gpus="V100-SXM2-16GB; " + str(plan.get("allocation_gpus_per_node", 4)) + " allocated per node",
                   mpi=result["mpi"],
                   url="https://github.com/Stardust0831/lammps/tree/sai-multigpu-reports/performance/"
                   + output.name,
                   image_sha256=plan["image_sha256"], binary_modified="-modified" in banner.group(1),
                   runtime_environment=plan.get("runtime_environment", {}),
                   reporter_sha256=digest(Path(__file__)),
                   sample_policy="Three sequential segments after warmup; SAI plots use medians and min/max ranges. "
                   "The upstream dashboard currently selects the best sample.", results={})
    if build_summary is not None:
        build = read(build_summary)
        if build["image_sha256"] != plan["image_sha256"] or build["commit"] != plan["source_sha"]:
            raise ValueError("build metadata belongs to a different image or source")
        summary["compiler"] = build["compiler"]
        summary["build_metadata_origin"] = build["action_url"]
        summary["build_metadata_sha256"] = digest(build_summary)
        write(output / "build-summary.json", build)
    summary["gpu_driver"] = ", ".join(sorted({d["driver"] for m in result["hardware"] for d in m["devices"]}))
    table = []
    for row in records:
        median = statistics.median(row["loop_seconds"])
        baseline = min((r for r in records if r["kind"] == row["kind"] and r["series"] == row["series"]),
                       key=lambda r: r["gpus"])
        base_time = statistics.median(baseline["loop_seconds"])
        speedup = base_time / median
        efficiency = speedup * baseline["gpus"] / row["gpus"] if row["series"] == "strong" else speedup
        table.append(dict(case=row["name"], series=row["series"], atoms=row["natoms"],
                          gpus=row["gpus"], nodes=row["nodes"], gpus_per_node=row.get("gpus_per_node", 4),
                          steps=row["steps"], median_seconds=median,
                          min_seconds=min(row["loop_seconds"]), max_seconds=max(row["loop_seconds"]),
                          ns_per_day=86400 / median * row["steps"] * (0.5e-6 if row["kind"] == "snap" else 2e-6),
                          relative_speedup=speedup, efficiency=efficiency,
                          full_process_seconds=row["wall_seconds"]))
        summary["results"][row["name"]] = dict(input=row["kind"], size=str(row["natoms"]),
              precision="double", package="kokkos", api="cuda", category="kokkos-cuda",
              flags="newton on neigh half comm device gpu/aware on "
              + " ".join(k + "=" + v for k, v in plan.get("runtime_environment", {}).items()), ranks=row["gpus"],
              threads=1, gpus=row["gpus"], natoms=row["natoms"], steps=row["steps"],
              nodes=row["nodes"], gpus_per_node=row.get("gpus_per_node", 4),
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
            selected = sorted((r for r in table if r["case"].startswith(kind + "-") and r["series"] == series),
                              key=lambda r: r["gpus"])
            ax = axes[rowid, colid]
            x = [r["gpus"] for r in selected]
            base = selected[0]["median_seconds"]
            if series == "strong":
                y = [r["relative_speedup"] for r in selected]
                lo = [r["relative_speedup"] - base / r["max_seconds"] for r in selected]
                hi = [base / r["min_seconds"] - r["relative_speedup"] for r in selected]
                ax.plot(x, [n / x[0] for n in x], "--", color="gray", label="Ideal")
                ax.set_ylabel("Speedup relative to 4 GPUs")
                ax.set_ylim(0, max(x) / min(x) * 1.06)
            else:
                y = [r["efficiency"] * 100 for r in selected]
                lo = [r["efficiency"] * 100 - base / r["max_seconds"] * 100 for r in selected]
                hi = [base / r["min_seconds"] * 100 - r["efficiency"] * 100 for r in selected]
                ax.axhline(100, ls="--", color="gray", label="Ideal")
                ax.set_ylabel("Weak-scaling efficiency (%)")
                ax.set_ylim(0, 105)
            ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", capsize=4, label="Median; min/max samples")
            ax.set_xticks(x, [str(r["gpus"]) + "\n" + str(r["nodes"]) + "x" + str(r["gpus_per_node"])
                             for r in selected])
            ax.set_xlabel("V100 GPUs (nodes x GPUs per node)")
            ax.set_title(("Tungsten SNAP" if kind == "snap" else "SPC/E water + PPPM") + " - " + series)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    setting = ", ".join(k + "=" + v for k, v in plan.get("runtime_environment", {}).items()) or "Site UCX settings"
    fig.suptitle("SAI LAMMPS Kokkos CUDA scaling - " + plan["source_sha"][:12] + "\n" + setting)
    fig.savefig(output / "scaling.png", dpi=180)
    fig.savefig(output / "scaling.svg")
    plt.close(fig)
    gpu_hours = float(accounting["ElapsedRaw"]) * plan["nodes"] * plan.get("allocation_gpus_per_node", 4) / 3600
    lines = ["# SAI LAMMPS multi-node performance trial", "",
             "![Scaling curves](scaling.png)", "",
             "Source: `" + plan["source_sha"] + "`. Accepted site image with existing DeePMD integration; "
             "the binary reports `-modified`. No LAMMPS source changes were made for this experiment.", "",
             "Slurm job `" + plan["job"] + "`: `" + accounting["State"] + "`, exit `" + accounting["ExitCode"]
             + "`, " + accounting["Elapsed"] + ", " + f"{gpu_hours:.2f} allocated GPU-hours. "
             "This includes reserved GPUs while smaller layouts run.", "",
             "Explicit runtime settings: `" + str(plan.get("runtime_environment", {})) + "`.", "",
             "One MPI rank per GPU, with the node layout recorded in the table, and one OpenMP thread per rank. "
             "MPI binds each rank to four physical cores, following the site's one-rank-per-GPU mapping. "
             "Three consecutive timed segments follow 100 warmup steps. Times exclude setup; "
             "full process wall times remain in the CSV and raw evidence. Nodes were not reserved exclusively.", "",
             "Kokkos CUDA runs asynchronously. The raw Pair/Comm/etc. sections can charge GPU waits "
             "to a later section and must not be read as network-only or kernel-only costs. "
             "The LAMMPS Kokkos guide recommends CUDA_LAUNCH_BLOCKING=1 for accurate section timings; "
             "that diagnostic mode was not used here because it changes execution performance. "
             "The curves use complete loop times.", "",
             "| Case | Atoms | Nodes x GPUs/node | Steps/sample | Median s | Min-max s | Strong speedup / weak efficiency |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(table, key=lambda r: (r["case"].split("-")[0], r["gpus"], r["series"])):
        metric = f"{row['relative_speedup']:.2f}x" if row["series"] == "strong" else f"{row['efficiency'] * 100:.1f}%"
        lines.append(f"| {row['case']} | {row['atoms']} | {row['nodes']}x{row['gpus_per_node']} | {row['steps']} | "
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


def compare_topologies(run, reference_run, output):
    """Compare identical input decks and image bytes across two allocations."""
    current, reference = read(run / "performance.json"), read(reference_run / "performance.json")
    if current["status"] != "PASS" or reference["status"] != "PASS":
        raise ValueError("both topology experiments must pass")
    if current["plan"].get("reference_sha256") != digest(reference_run / "performance.json"):
        raise ValueError("comparison is not the frozen reference experiment")
    for key in ("image_sha256", "source_sha", "recipe_sha256", "partition"):
        if current["plan"][key] != reference["plan"][key]:
            raise ValueError("comparison differs in " + key)
    if current["mpi"] != reference["mpi"]:
        raise ValueError("comparison used a different host MPI version")
    for key in ("cpu",):
        if {r[key] for r in current["hardware"]} != {r[key] for r in reference["hardware"]}:
            raise ValueError("comparison used different hardware")
    for key in ("name", "driver"):
        if ({d[key] for r in current["hardware"] for d in r["devices"]}
                != {d[key] for r in reference["hardware"] for d in r["devices"]}):
            raise ValueError("comparison used different GPU model or driver")
    old = {r["name"]: r for r in reference["measurements"]}
    rows = []
    for after in current["measurements"]:
        before = old[after["name"]]
        for key in ("kind", "series", "dims", "steps", "warmup", "repeats", "natoms", "gpus"):
            if before[key] != after[key]:
                raise ValueError("comparison changed " + key)
        for filename in ("in.lmp", "W.snapcoeff", "W.snapparam") if after["kind"] == "snap" else ("in.lmp", "data.spce"):
            if digest(run / "cases" / after["name"] / filename) != digest(reference_run / "cases" / after["name"] / filename):
                raise ValueError("comparison changed the actual input: " + filename)
        delta = abs(before["states"]["initial"]["values"][0] - after["states"]["initial"]["values"][0])
        if not math.isfinite(delta) or delta > 1e-6:
            raise ValueError("initial potential energy differs between experiments")
        rows.append(dict(case=after["name"], kind=after["kind"], series=after["series"], gpus=after["gpus"],
                         previous_layout=f"{before['nodes']}x{before.get('gpus_per_node', 4)}",
                         current_layout=f"{after['nodes']}x{after.get('gpus_per_node', 4)}",
                         previous_seconds=before["loop_seconds"], current_seconds=after["loop_seconds"],
                         previous_median=statistics.median(before["loop_seconds"]),
                         current_median=statistics.median(after["loop_seconds"]),
                         speedup=statistics.median(before["loop_seconds"]) / statistics.median(after["loop_seconds"]),
                         initial_pe_per_atom_difference=delta))
    limitations = ("Independent allocations at different times, with different node placement and contention. "
                  "The current allocation reserves all 16 GPUs per node, including during smaller cases; "
                  "the reference reserves four per node. The current run also verifies CPU-local GPU selection; "
                  "the reference used allocated GPU index order and did not record NUMA locality. "
                  "Runtime environment overrides are recorded separately for both experiments. "
                  "Ratios include these effects and do not isolate node count alone. "
                  "Error bars show the range of three sequential samples, not confidence intervals.")
    write(output / "topology-comparison.json", dict(reference_job=reference["plan"]["job"],
          current_job=current["plan"]["job"], limitations=limitations, results=rows,
          reference_environment=reference["plan"].get("runtime_environment", {}),
          current_environment=current["plan"].get("runtime_environment", {})))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for i, kind in enumerate(("snap", "spce")):
        for j, series in enumerate(("strong", "weak")):
            selected = sorted((r for r in rows if r["kind"] == kind and r["series"] == series), key=lambda r: r["gpus"])
            ax = axes[i, j]
            labels = []
            for prefix, experiment, layout in (("previous", reference, "4 GPUs/node"), ("current", current, "Up to 16 GPUs/node")):
                environment = experiment["plan"].get("runtime_environment", {})
                label = "CUDA IPC disabled" if environment.get("UCX_TLS") == "^cuda_ipc" else "site UCX settings"
                if environment and environment.get("UCX_TLS") != "^cuda_ipc":
                    label = ", ".join(k + "=" + v for k, v in environment.items())
                labels.append((prefix, layout + "; " + label))
            for prefix, label in labels:
                medians = [r[prefix + "_median"] for r in selected]
                low = [r[prefix + "_median"] - min(r[prefix + "_seconds"]) for r in selected]
                high = [max(r[prefix + "_seconds"]) - r[prefix + "_median"] for r in selected]
                ax.errorbar([r["gpus"] for r in selected], medians, yerr=[low, high], fmt="o-", capsize=4, label=label)
            ax.set_xticks([r["gpus"] for r in selected])
            ax.set_xlabel("Total V100 GPUs")
            ax.set_ylabel("Seconds per measured segment")
            ax.set_title(("Tungsten SNAP" if kind == "snap" else "SPC/E water + PPPM") + " - " + series)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("SAI LAMMPS node layout and communication settings - " + current["plan"]["source_sha"][:12])
    fig.savefig(output / "topology-comparison.png", dpi=180)
    fig.savefig(output / "topology-comparison.svg")
    plt.close(fig)
    lines = ["", "## Comparison with the four-GPU-per-node trial", "",
             "![Node layout comparison](topology-comparison.png)", "", limitations, "",
             "The SIF hash, source, compiler recipe, host MPI version, GPU/CPU models and GPU driver match. "
             "Every compared input deck and model/data file is byte-identical. "
             "Initial potential energies agree within 1e-6 per atom.", "",
             "| Case | Previous layout | Current layout | Previous median s | Current median s | Ratio |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["kind"], r["gpus"], r["series"])):
        lines.append(f"| {r['case']} | {r['previous_layout']} | {r['current_layout']} | "
                     f"{r['previous_median']:.3f} | {r['current_median']:.3f} | {r['speedup']:.3f}x |")
    lines += ["", "A ratio above one means the current run was faster. "
              "The 1x4 repeat is a baseline check for allocation-to-allocation variation.", "",
              "[Exact comparison samples](topology-comparison.json)", ""]
    with (output / "README.md").open("a") as stream:
        stream.write("\n".join(lines))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-summary", type=Path, help="Earlier build metadata with the identical image checksum")
    parser.add_argument("--reference-run", type=Path, help="Archived reference run for the topology comparison")
    args = parser.parse_args()
    build_report(args.run, read(args.accounting), args.output, args.build_summary)
    if args.reference_run is not None:
        compare_topologies(args.run, args.reference_run, args.output)
