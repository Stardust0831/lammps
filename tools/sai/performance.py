#!/usr/bin/env python3
"""Measured SNAP/SPC-E scaling, using an immutable accepted image under Slurm."""
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

from common import digest, read, stamp, write
from suite import compare, execute, node_info, rank


def layout_sizes(layout):
    """Old plans list node counts; new plans also specify ranks per node."""
    nodes, per_node = (layout, 4) if isinstance(layout, int) else (layout["nodes"], layout["gpus_per_node"])
    if nodes not in (1, 2, 4, 8) or per_node not in (4, 8, 16):
        raise ValueError("unsupported GPU layout")
    return nodes, per_node


def cpu_list(text):
    cpus = set()
    for item in text.strip().split(","):
        ends = item.split("-")
        if len(ends) > 2 or not all(v.isdigit() for v in ends):
            raise ValueError("invalid CPU locality list")
        first, last = int(ends[0]), int(ends[-1])
        if first > last:
            raise ValueError("reversed CPU locality range")
        cpus.update(range(first, last + 1))
    return sorted(cpus)


def local_device_order(devices, locality):
    """Match sequential MPI core binding to GPU CPU-locality groups."""
    result = [dict(device, local_cpus=cpu_list(locality[device["index"]])) for device in devices]
    return sorted(result, key=lambda device: (min(device["local_cpus"]), int(device["index"])))


def input_text(kind, dims, backend, steps=100, warmup=100, repeats=3, seed=None, gate=False):
    """Explicit device styles prevent suffix fallback for measured operations."""
    suffix = "/kk" if backend == "kokkos" else ""
    data = Path(".")
    lines = ["units " + ("metal" if kind == "snap" else "real"),
             "atom_style " + ("atomic" if kind == "snap" else "full"),
             "boundary p p p", "newton on", "atom_modify map array"]
    if seed:
        lines += ["read_data " + Path(seed).name]
    elif kind == "snap":
        lines += ["lattice bcc 3.1803 origin 0.1 0.1 0.1",
                  "region box block 0 %d 0 %d 0 %d" % tuple(dims),
                  "create_box 1 box", "create_atoms 1 box", "mass 1 183.84",
                  "displace_atoms all random 0.001 0.001 0.001 87287 units box"]
    else:
        lines += ["read_data " + str(data / "data.spce"), "replicate %d %d %d" % tuple(dims)]
    if kind == "snap":
        lines += ["pair_style hybrid/overlay" + suffix + " zbl" + suffix + " 4.0 4.8 snap" + suffix,
                  "pair_coeff 1 1 zbl" + suffix + " 74 74",
                  "pair_coeff * * snap" + suffix + " " + str(data / "W.snapcoeff")
                  + " " + str(data / "W.snapparam") + " W",
                  "neighbor 0.0 bin", "neigh_modify every 1 delay 0 check yes"]
    else:
        lines += ["pair_style lj/cut/coul/long" + suffix + " 9.8 9.8",
                  "pair_coeff 1 1 0.15535 3.166", "pair_coeff * 2 0.0 0.0",
                  "kspace_style pppm" + suffix + " 1e-4",
                  "bond_style harmonic" + suffix, "angle_style harmonic" + suffix,
                  "bond_coeff 1 1000.0 1.0", "angle_coeff 1 100.0 109.47",
                  "special_bonds lj/coul 0.0 0.0 0.5", "neighbor 2.0 bin",
                  "neigh_modify every 1 delay 0 check yes",
                  "fix constraints all shake" + suffix + " 0.0001 20 0 b 1 a 1"]
    temperature = 800 if kind == "snap" else 300
    if not seed:
        # Geometry-based RNG is independent of MPI-assigned atom IDs in created lattices.
        lines += [f"velocity all create {temperature} 432567 dist gaussian loop geom"]
    if gate:
        lines += ["fix integrate all nve" + suffix]
    else:
        damping = "0.025" if kind == "snap" else "100.0"
        lines += [f"fix integrate all nvt{suffix} temp {temperature} {temperature} {damping}"]
    lines += ["timestep " + ("0.0005" if kind == "snap" else "0.5" if gate else "2.0"),
              "thermo 100", "thermo_style custom step atoms temp pe ke etotal press pxx pyy pzz pxy pxz pyz",
              "thermo_modify norm no format float %.16g", "run 0"]

    def state(label):
        return ['print "BENCH_STATE ' + label + ' $(atoms:%.0f) '
                + " ".join("$(" + v + ":%.16g)" for v in
                           ("pe/atoms", "ke/atoms", "etotal/atoms", "temp", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")) + '"']

    lines += state("initial")
    if gate:
        if not seed:
            lines += ["write_data seed.data nocoeff"]
        lines += ["write_dump all custom initial.dump id fx fy fz modify sort id format float %.17g",
                  "run 10", "write_dump all custom final.dump id fx fy fz modify sort id format float %.17g"]
    else:
        lines += [f"run {warmup}"]
        for sample in range(repeats):
            lines += [f'print "BENCH_SAMPLE {sample}"', f"run {steps}"]
    lines += state("final")
    return "\n".join(lines) + "\n"


def states(text):
    out = {}
    for label, natoms, values in re.findall(r"^BENCH_STATE (initial|final) (\d+) (.+)$", text, re.M):
        numbers = list(map(float, values.split()))
        if label in out or len(numbers) != 10 or not all(map(math.isfinite, numbers)):
            raise ValueError("invalid thermodynamic state")
        out[label] = dict(natoms=int(natoms), values=numbers)
    if set(out) != {"initial", "final"} or out["initial"]["natoms"] != out["final"]["natoms"]:
        raise ValueError("missing states or changed atom count")
    return out


def dump_forces(path, natoms):
    lines = Path(path).read_text().splitlines()
    start = lines.index("ITEM: ATOMS id fx fy fz") + 1
    atoms = {}
    for line in lines[start:]:
        fields = line.split()
        atom = int(fields[0])
        values = list(map(float, fields[1:]))
        if atom in atoms or len(values) != 3 or not all(map(math.isfinite, values)):
            raise ValueError("invalid force dump")
        atoms[atom] = values
    if set(atoms) != set(range(1, natoms + 1)):
        raise ValueError("incomplete force dump")
    return [x for atom in sorted(atoms) for x in atoms[atom]]


def measured_loops(text, steps, repeats, natoms):
    pattern = r"Loop time of ([\d.eE+-]+) on \d+ procs for (\d+) steps with (\d+) atoms"
    chunks = re.split(r"^BENCH_SAMPLE \d+\s*$", text, flags=re.M)[1:]
    if len(chunks) != repeats:
        raise ValueError("missing measured samples")
    times = []
    for chunk in chunks:
        matches = re.findall(pattern, chunk)
        if len(matches) != 1:
            raise ValueError("missing or duplicate measured loop")
        seconds, count, atoms = matches[0]
        seconds = float(seconds)
        if int(count) != steps or int(atoms) != natoms or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("unexpected loop size or timing")
        times.append(seconds)
    return times


def monitor(run):
    node = socket.gethostname().split(".")[0]
    allocation = read(Path(run) / "nodes" / (node + ".json"))
    mask = ",".join(row["uuid"] for row in allocation["devices"])
    fields = "timestamp,uuid,memory.used,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu"
    with (Path(run) / "nodes" / (node + "-telemetry.csv")).open("w") as out:
        while not (Path(run) / "telemetry.stop").exists():
            subprocess.run(["nvidia-smi", "-i", mask, "--query-gpu=" + fields,
                            "--format=csv,noheader,nounits"], stdout=out, stderr=subprocess.STDOUT)
            out.flush()
            time.sleep(2)


def run_benchmarks(directory):
    run = Path(directory).resolve()
    plan = read(run / "plan.json")
    records, gates = [], []
    telemetry = None
    result = dict(plan=plan, started_at=stamp(), status="RUNNING", measurements=records, numerical_checks=gates)
    write(run / "performance.json", result)
    try:
        if os.environ.get("SLURM_JOB_ID") != plan["job"]:
            raise ValueError("not in the assigned Slurm job")
        for key, value in plan.get("runtime_environment", {}).items():
            if os.environ.get(key) != value:
                raise ValueError("wrong runtime setting: " + key)
        if os.environ["SAI_SIF_LAMMPS_SOURCE"] != plan["source_sha"]:
            raise ValueError("wrong source module")
        if digest(os.environ["SAI_SIF_LAMMPS_IMAGE"]) != plan["image_sha256"]:
            raise ValueError("image checksum changed")
        for name, entry in read(run / "control/benchmarks/sources.json").items():
            if digest(run / "control/benchmarks" / name) != entry["sha256"]:
                raise ValueError("input checksum changed: " + name)
        count = plan["nodes"]
        allocated_per_node = plan.get("allocation_gpus_per_node", 4)
        srun = ["srun", f"--nodes={count}", f"--ntasks={count}", "--ntasks-per-node=1",
                f"--gres=gpu:{allocated_per_node}", "--export=ALL", "/usr/bin/python3", "-B",
                str(run / "control/performance.py")]
        execute(srun + ["node", str(run)], run / "preflight.log")
        machines = [read(p) for p in sorted((run / "nodes").glob("*.json"))]
        nodes = [m["node"] for m in machines]
        if len(nodes) != count or len({m["cpu"] for m in machines}) != 1:
            raise ValueError("missing nodes or different CPU models")
        if len({d["name"] for m in machines for d in m["devices"]}) != 1:
            raise ValueError("different GPU models")
        result["hardware"] = machines
        result["mpi"] = subprocess.check_output(["mpirun", "--version"], text=True).strip()
        telemetry = subprocess.Popen([srun[0], "--overlap", *srun[1:], "monitor", str(run)], stdout=(run / "telemetry.log").open("w"),
                                     stderr=subprocess.STDOUT)
        lmp = shutil.which("lmp")

        def launch(name, kind, dims, count, steps=100, warmup=100, repeats=3, backend="kokkos", seed=None, gate=False, per_node=4):
            per_node = 1 if backend == "cpu" else per_node
            if count > len(nodes) or per_node > allocated_per_node:
                raise ValueError("layout exceeds allocation")
            case = run / "cases" / name
            case.mkdir(parents=True)
            for filename in (("W.snapcoeff", "W.snapparam") if kind == "snap" else ("data.spce",)):
                shutil.copyfile(run / "control/benchmarks" / filename, case / filename)
            if seed:
                shutil.copyfile(seed, case / Path(seed).name)
            (case / "in.lmp").write_text(input_text(kind, dims, backend, steps, warmup, repeats, seed, gate))
            selected = nodes[:count]
            args = [] if backend == "cpu" else ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                    "neigh", "half", "newton", "on", "comm", "device", "gpu/aware", "on"]
            args += ["-in", "in.lmp", "-log", "log.lammps"]
            argv = ["mpirun", "-np", str(count * per_node), "--host",
                    ",".join(n + ":" + str(per_node) for n in selected), "--map-by",
                    "ppr:" + str(per_node) + ":node:pe=4", "--bind-to", "core", "/usr/bin/python3", "-B",
                    str(run / "control/performance.py"), "rank", str(run), str(case), lmp, *args]
            write(case / "command.json", argv)
            wall = execute(argv, case / "screen.txt", cwd=case, timeout=1200,
                           fatal_pattern=r"\bUCX\s+ERROR\b|cuIpcOpenMemHandle\(\) failed")
            text = (case / "screen.txt").read_text()
            if backend == "kokkos":
                if "KOKKOS mode with" not in text or "/kk" not in text:
                    raise ValueError("missing CUDA execution evidence")
                if re.search(r"WARNING:.*(fallback|fall back|not compatible)", text, re.I):
                    raise ValueError("accelerator fallback warning")
            state = states(text)
            expected = 2 * math.prod(dims) if kind == "snap" else 4500 * math.prod(dims)
            if state["initial"]["natoms"] != expected:
                raise ValueError("wrong atom count")
            ranks = [read(p) for p in sorted((case / "ranks").glob("*.json"))]
            if sorted(r["rank"] for r in ranks) != list(range(count * per_node)):
                raise ValueError("incomplete MPI rank evidence")
            for node in selected:
                rows = [r for r in ranks if r["node"] == node]
                if (len(rows) != per_node or len({r["gpu"]["uuid"] for r in rows}) != per_node
                        or len(set().union(*(set(r["affinity"]) for r in rows)))
                        != sum(len(r["affinity"]) for r in rows)):
                    raise ValueError("GPU or CPU rank mapping overlaps")
            return case, text, state, wall

        for kind in ("snap", "spce"):
            dims = (8, 8, 8) if kind == "snap" else (1, 1, 1)
            fixture, _, _, _ = launch(kind + "-fixture", kind, dims, 1, backend="cpu", gate=True)
            seed = fixture / "seed.data"
            ref, _, ref_state, _ = launch(kind + "-cpu-reference", kind, dims, 1,
                                          backend="cpu", seed=seed, gate=True)
            tolerance = dict(atol=2e-6, rtol=2e-7)
            for layout in plan["layouts"]:
                count, per_node = layout_sizes(layout)
                case, _, state, wall = launch(f"{kind}-gate-{per_node * count}gpu", kind, dims, count,
                                             seed=seed, gate=True, per_node=per_node)
                deltas = {}
                for phase in ("initial", "final"):
                    deltas[phase + "_thermo_max"] = compare(ref_state[phase]["values"], state[phase]["values"], **tolerance)
                    natoms = state[phase]["natoms"]
                    deltas[phase + "_force_max"] = compare(dump_forces(ref / (phase + ".dump"), natoms),
                                                           dump_forces(case / (phase + ".dump"), natoms), **tolerance)
                gates.append(dict(case=case.name, status="PASS", nodes=count, gpus_per_node=per_node,
                                  tolerance=tolerance, differences=deltas, wall_seconds=wall))
                write(run / "performance.json", result)
                print("NUMERICAL", case.name, deltas, flush=True)

        for spec in plan["benchmarks"]:
            kind, dims, count = spec["kind"], spec["dims"], spec["nodes"]
            per_node = spec.get("gpus_per_node", 4)
            name = spec["name"]
            print("MEASURING", name, dims, count * per_node, flush=True)
            case, text, state, wall = launch(name, kind, dims, count, steps=spec["steps"],
                                            warmup=spec["warmup"], repeats=spec["repeats"], per_node=per_node)
            natoms = state["initial"]["natoms"]
            timings = measured_loops(text, spec["steps"], spec["repeats"], natoms)
            temperature = state["final"]["values"][3]
            if not (400 < temperature < 1200 if kind == "snap" else 150 < temperature < 450):
                raise ValueError("temperature outside predeclared stability bounds")
            records.append(dict(spec, status="PASS", natoms=natoms, gpus=count * per_node,
                                loop_seconds=timings, median_seconds=statistics.median(timings),
                                atom_steps_per_second=[natoms * spec["steps"] / t for t in timings],
                                wall_seconds=wall, states=state,
                                timing_sections=re.findall(r"^(?:Pair|Bond|Kspace|Neigh|Comm|Modify|Output|Other)\s.*$", text, re.M)))
            write(run / "performance.json", result)
            print("MEASURED", name, "atoms", natoms, "seconds", timings, flush=True)
        result["status"] = "PASS"
    except Exception as exc:
        result.update(status="FAIL", error=str(exc))
        print("FAILED", str(exc), flush=True)
    finally:
        (run / "telemetry.stop").touch()
        if telemetry:
            try:
                telemetry.wait(timeout=15)
            except subprocess.TimeoutExpired:
                telemetry.terminate()
                telemetry.wait(timeout=10)
        result["finished_at"] = stamp()
        write(run / "performance.json", result)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if sys.argv[1] == "node":
        directory = Path(sys.argv[2])
        plan = read(directory / "plan.json")
        node_info(directory, plan.get("allocation_gpus_per_node", 4))
        node = socket.gethostname().split(".")[0]
        for label, argv in (("gpu-topology", ["nvidia-smi", "topo", "-m"]),
                            ("cpu-topology", ["lscpu", "-e=CPU,CORE,SOCKET,NODE"])):
            execute(argv, directory / "nodes" / (node + "-" + label + ".txt"))
        if plan.get("gpu_assignment") == "cpu-local":
            buses = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id",
                                             "--format=csv,noheader,nounits"], text=True)
            locality = {}
            for line in buses.splitlines():
                index, bus_id = (part.strip() for part in line.split(","))
                domain, bus, device = bus_id.lower().split(":")
                address = f"{int(domain, 16):04x}:{bus}:{device}"
                locality[index] = (Path("/sys/bus/pci/devices") / address / "local_cpulist").read_text()
            path = directory / "nodes" / (node + ".json")
            info = read(path)
            info["allocated_device_order"] = [d["index"] for d in info["devices"]]
            info["devices"] = local_device_order(info["devices"], locality)
            write(path, info)
    elif sys.argv[1] == "monitor":
        monitor(sys.argv[2])
    elif sys.argv[1] == "rank":
        directory = Path(sys.argv[2])
        if read(directory / "plan.json").get("gpu_assignment") == "cpu-local":
            node = socket.gethostname().split(".")[0]
            info = read(directory / "nodes" / (node + ".json"))
            device = info["devices"][int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])]
            if not set(os.sched_getaffinity(0)).issubset(device["local_cpus"]):
                raise RuntimeError("MPI CPU binding does not match selected GPU locality")
        rank(*sys.argv[2:5], sys.argv[5:])
    elif sys.argv[1] == "run":
        raise SystemExit(run_benchmarks(sys.argv[2]))
    else:
        raise SystemExit("expected node, monitor, rank or run")
