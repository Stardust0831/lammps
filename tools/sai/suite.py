#!/usr/bin/env python3
"""Run deterministic CPU/GPU comparisons inside a two-node Slurm allocation."""
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import socket
import subprocess
import sys
import time

from common import digest, read, reports, write

MODES = (("cpu-1", "cpu", 1, 1), ("gpu-1", "gpu", 1, 1),
         ("gpu-2", "gpu", 1, 2), ("gpu-4-2nodes", "gpu", 2, 2),
         ("kokkos-1", "kokkos", 1, 1), ("kokkos-2", "kokkos", 1, 2),
         ("kokkos-4-2nodes", "kokkos", 2, 2))
ATOMS = 2048


def execute(argv, output, *, cwd=None, timeout=240):
    started = time.monotonic()
    with Path(output).open("w") as stream:
        process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT,
                                   cwd=cwd, start_new_session=True)
        try:
            result = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError("command timed out; see " + str(output)) from None
    if result:
        tail = Path(output).read_text(errors="replace")[-1600:]
        raise RuntimeError("command exited " + str(result) + ": " + tail)
    return time.monotonic() - started


def node_info(run):
    """One Slurm step per node records its own allocated GPU mask."""
    node = socket.gethostname().split(".")[0]
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not mask or mask == "-1":
        raise RuntimeError("Slurm did not expose GPUs on " + node)
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                                     "--format=csv,noheader,nounits"], text=True)
    devices = []
    inventory = [[field.strip() for field in row] for row in csv.reader(lines.splitlines())]
    for visible in mask.split(","):
        matches = [row for row in inventory if row[0] == visible or row[1] == visible]
        if len(matches) != 1:
            raise RuntimeError("cannot resolve allocated GPU " + visible)
        index, uuid, name, driver = matches[0]
        devices.append(dict(index=index, uuid=uuid, name=name, driver=driver))
    if len(devices) != 2:
        raise RuntimeError("expected exactly two allocated GPUs per node")
    cpu = subprocess.check_output(["lscpu", "-J"], text=True)
    fields = {r["field"].rstrip(":"): r["data"] for r in json.loads(cpu)["lscpu"]}
    write(Path(run) / "nodes" / (node + ".json"), dict(node=node, devices=devices,
          cpu=fields["Model name"], logical_cpus=fields["CPU(s)"],
          affinity=sorted(os.sched_getaffinity(0)), operating_system=" ".join(os.uname())))


def rank(run, case, program, arguments):
    run, case = Path(run), Path(case)
    node = socket.gethostname().split(".")[0]
    allocation = read(run / "nodes" / (node + ".json"))
    local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    global_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    selected = allocation["devices"][local_rank]
    # UUIDs remain stable even when Slurm assigns different indices on each node.
    os.environ["CUDA_VISIBLE_DEVICES"] = selected["uuid"]
    write(case / "ranks" / (str(global_rank) + ".json"), dict(
        node=node, rank=global_rank, local_rank=local_rank, gpu=selected,
        affinity=sorted(os.sched_getaffinity(0)), cuda_visible_devices=selected["uuid"],
        source_sha=os.environ["SAI_SIF_LAMMPS_SOURCE"],
        image=os.environ["SAI_SIF_LAMMPS_IMAGE"], program=program))
    os.execv(program, [program, *arguments])


def fixture():
    """Perturbed periodic FCC system, explicit velocities and neutral charges."""
    rng = random.Random(93541)
    lattice = (4 / 0.7) ** (1 / 3)
    box = 8 * lattice
    basis = ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))
    rows, velocities = [], []
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for offset in basis:
                    atom = len(rows) + 1
                    xyz = [((cell + frac) * lattice + rng.uniform(-.025, .025)) % box
                           for cell, frac in zip((i, j, k), offset)]
                    rows.append(f"{atom} {1 + atom % 2} {0.5 if atom % 2 else -0.5} "
                                + " ".join(format(x, ".17g") for x in xyz))
                    velocities.append(f"{atom} " + " ".join(format(rng.uniform(-.1, .1), ".17g")
                                                           for _ in range(3)))
    return ("SAI deterministic periodic FCC test\n\n2048 atoms\n2 atom types\n\n"
            + "\n".join(f"0 {box:.17g} {axis}lo {axis}hi" for axis in "xyz")
            + "\n\nAtoms # charge\n\n" + "\n".join(rows)
            + "\n\nVelocities\n\n" + "\n".join(velocities) + "\n")


def deck(kind, backend):
    suffix = {"cpu": "", "gpu": "/gpu", "kokkos": "/kk"}[backend]
    pair = "lj/cut" if kind == "lj" else "lj/cut/coul/long"
    lines = ["units lj", "atom_style charge", "boundary p p p", "newton off",
             "atom_modify map array", "read_data data.lmp", "mass * 1.0",
             "pair_style " + pair + suffix + " 2.5", "pair_coeff * * 1.0 1.0",
             "neighbor 0.3 bin", "neigh_modify every 1 delay 0 check yes"]
    if kind == "pppm":
        lines += ["kspace_style pppm" + suffix + " 1e-6",
                  "kspace_modify mesh 32 32 32 order 5"]
    lines += ["thermo 10", "thermo_style custom step atoms pe ke etotal pxx pyy pzz pxy pxz pyz",
              "thermo_modify norm no format float %.17g", "timestep 0.001", "fix integrate all nve"]
    for label, steps in (("initial", 0), ("final", 20)):
        lines += ["run " + str(steps), 'print "SAI_VALUES ' + label + ' $(atoms:%.0f) '
                  + " ".join("$(" + x + ":%.17g)" for x in ("pe", "ke", "etotal", "pxx", "pyy", "pzz", "pxy", "pxz", "pyz")) + '"',
                  "write_dump all custom " + label + ".dump id x y z fx fy fz "
                  "modify sort id format float %.17g"]
    return "\n".join(lines) + "\n"


def values(path):
    text = Path(path).read_text()
    records = {}
    for name, atoms, rest in re.findall(r"^SAI_VALUES (initial|final) (\d+) (.+)$", text, re.M):
        numbers = list(map(float, rest.split()))
        if name in records or int(atoms) != ATOMS or len(numbers) != 9 or not all(map(math.isfinite, numbers)):
            raise ValueError("invalid or duplicate thermodynamic output")
        records[name] = numbers
    if set(records) != {"initial", "final"}:
        raise ValueError("missing thermodynamic output")
    return records


def forces(path):
    lines = Path(path).read_text().splitlines()
    start = lines.index("ITEM: ATOMS id x y z fx fy fz") + 1
    data = {}
    for line in lines[start:]:
        fields = line.split()
        atom = int(fields[0])
        numbers = list(map(float, fields[1:]))
        if atom in data or len(numbers) != 6 or not all(map(math.isfinite, numbers)):
            raise ValueError("invalid atom output")
        data[atom] = numbers
    if set(data) != set(range(1, ATOMS + 1)):
        raise ValueError("missing or duplicate atom IDs")
    return data


def compare(reference, actual, *, atol, rtol):
    if len(reference) != len(actual):
        raise AssertionError("different vector lengths")
    maximum = 0.0
    for a, b in zip(reference, actual):
        if not math.isfinite(a) or not math.isfinite(b):
            raise AssertionError("non-finite result")
        error = abs(a - b)
        maximum = max(maximum, error)
        if error > atol + rtol * abs(a):
            raise AssertionError(f"difference {error:.8g} exceeds {atol} + {rtol} * |{a:.8g}|")
    return maximum


def verify(case, reference, backend, nodes, per_node):
    result = values(case / "screen.txt")
    reference_values = values(reference / "screen.txt")
    # GPU mixed precision needs a finite absolute tolerance near zero.
    tolerance = dict(atol=2e-5, rtol=2e-5) if backend == "gpu" else dict(atol=2e-8, rtol=2e-8)
    deltas = {}
    for phase in ("initial", "final"):
        deltas[phase + "_thermo_max"] = compare(reference_values[phase], result[phase], **tolerance)
        actual_atoms, expected_atoms = forces(case / (phase + ".dump")), forces(reference / (phase + ".dump"))
        deltas[phase + "_force_max"] = compare(
            [v for atom in sorted(expected_atoms) for v in expected_atoms[atom][3:]],
            [v for atom in sorted(actual_atoms) for v in actual_atoms[atom][3:]], **tolerance)
    drift = abs(result["final"][2] - result["initial"][2]) / ATOMS
    if drift > 1e-4:
        raise AssertionError(f"NVE energy drift {drift} per atom exceeds 1e-4")
    ranks = [read(p) for p in sorted((case / "ranks").glob("*.json"))]
    if sorted(row["rank"] for row in ranks) != list(range(len(nodes) * per_node)):
        raise AssertionError("missing or duplicate MPI ranks")
    for node in nodes:
        selected = [row for row in ranks if row["node"] == node]
        if len(selected) != per_node or len({r["gpu"]["uuid"] for r in selected}) != per_node:
            raise AssertionError("MPI ranks do not map to distinct allocated GPUs")
    screen = (case / "screen.txt").read_text()
    if backend == "gpu" and "GPU Time Info" not in screen:
        raise AssertionError("missing actual GPU-package execution evidence")
    if backend == "kokkos" and not re.search(r"KOKKOS mode.*enabled", screen, re.I):
        raise AssertionError("missing Kokkos execution evidence")
    return dict(max_errors=deltas, tolerance=tolerance, energy_drift_per_atom=drift,
                nodes=nodes, ranks=ranks)


def run_suite(run):
    run = Path(run).resolve()
    plan = read(run / "plan.json")
    cases, machine, banner, version = [], {}, "", ""
    if os.environ.get("SLURM_JOB_ID") != plan["job"]:
        raise RuntimeError("not inside the assigned Slurm job")
    try:
        if os.environ["SAI_SIF_LAMMPS_SOURCE"] != plan["source_sha"]:
            raise RuntimeError("loaded module has a different source commit")
        if digest(os.environ["SAI_SIF_LAMMPS_IMAGE"]) != plan["image_sha256"]:
            raise RuntimeError("accepted image bytes changed")
        execute(["srun", "--nodes=2", "--ntasks=2", "--ntasks-per-node=1", "--gres=gpu:2",
                 "--export=ALL", "/usr/bin/python3", "-B", str(run / "control/suite.py"),
                 "node", str(run)], run / "preflight.log")
        machines = [read(p) for p in sorted((run / "nodes").glob("*.json"))]
        if len(machines) != 2:
            raise RuntimeError("expected two distinct compute nodes")
        nodes = [m["node"] for m in machines]
        machine = dict(nodes=machines, operating_system=machines[0]["operating_system"],
                       mpi=subprocess.check_output(["mpirun", "--version"], text=True).strip(),
                       threads_per_rank=1, partition=plan["partition"])
        lmp = subprocess.check_output(["which", "lmp"], text=True).strip()

        def command(case, selected, per_node, arguments):
            return ["mpirun", "-np", str(len(selected) * per_node), "--host",
                    ",".join(n + ":" + str(per_node) for n in selected),
                    "--map-by", "ppr:" + str(per_node) + ":node", "--bind-to", "none",
                    "/usr/bin/python3", "-B", str(run / "control/suite.py"), "rank", str(run),
                    str(case), lmp, *arguments]

        help_case = run / "help"
        help_case.mkdir()
        execute(command(help_case, nodes[:1], 1, ["-h"]), run / "help.txt", cwd=help_case)
        help_text = (run / "help.txt").read_text()
        match = re.search(r"Git info \([^\n]+\)", help_text)
        banner = match.group(0) if match else ""
        if not banner or not any(plan["source_sha"][:n] in banner for n in range(7, 41)):
            raise RuntimeError("executable Git banner does not match the requested develop commit")
        version = banner
        compiler = re.search(r"^Compiler:.*$", help_text, re.M)
        if compiler:
            machine["compiler"] = compiler.group(0)
        cases.append(dict(name="provenance", status="pass", seconds=0, message=banner))
        inputs = run / "inputs"
        inputs.mkdir()
        (inputs / "data.lmp").write_text(fixture())
        for kind in ("lj", "pppm"):
            reference = run / "cases" / (kind + "-cpu-1")
            for label, backend, count, per_node in MODES:
                name = kind + "-" + label
                case = run / "cases" / name
                case.mkdir(parents=True)
                (case / "data.lmp").write_text((inputs / "data.lmp").read_text())
                (case / "in.lmp").write_text(deck(kind, backend))
                arguments = []
                if backend == "gpu":
                    arguments = ["-pk", "gpu", "1", "neigh", "yes", "split", "1.0"]
                if backend == "kokkos":
                    arguments = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "neigh", "full",
                                 "comm", "device", "gpu/aware", "on"]
                arguments += ["-in", "in.lmp", "-log", "log.lammps"]
                argv = command(case, nodes[:count], per_node, arguments)
                write(case / "command.json", argv)
                started = time.monotonic()
                record = dict(name=name, status="pass", message="")
                try:
                    execute(argv, case / "screen.txt", cwd=case)
                    record.update(verify(case, reference, backend, nodes[:count], per_node))
                except AssertionError as exc:
                    record.update(status="failure", message=str(exc))
                except Exception as exc:
                    record.update(status="error", message=str(exc))
                record["seconds"] = round(time.monotonic() - started, 6)
                cases.append(record)
                print(name, record["status"], record["message"], flush=True)
    except Exception as exc:
        cases.append(dict(name="infrastructure", status="error", seconds=0, message=str(exc)))
    result = reports(run, cases, machine, version, banner)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    if sys.argv[1] == "node":
        node_info(sys.argv[2])
    elif sys.argv[1] == "rank":
        rank(*sys.argv[2:5], sys.argv[5:])
    elif sys.argv[1] == "run":
        raise SystemExit(run_suite(sys.argv[2]))
    else:
        raise SystemExit("unknown operation")
