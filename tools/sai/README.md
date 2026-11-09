# SAI external MPI/GPU regression tester

This fork runs a small, reproducible end-to-end GPU regression suite at
[SAI](https://www.open-sai.com/). A GitHub self-hosted runner on the operator's
VM submits Slurm jobs and publishes results. Compilation and computation stay
on SAI; the VM does not compile LAMMPS or store container images.

The tested binary is the accepted daily LAMMPS image from
[sai-hpc-software](https://github.com/AI4SAI/sai-hpc-software). The tester resolves
the current upstream `develop` SHA and waits until the accepted image matches
it. The exact image, recipe and controller hashes are recorded. The image
includes the site's existing DeePMD integration; this tester does not modify
LAMMPS sources or rebuild dependencies.

## Tests

One job requests two nodes with two V100 GPUs per node in the `16V100`
partition. Host Open MPI launches the accepted executable through its existing
Apptainer wrapper. Every rank records its node, CPU affinity and allocated GPU
UUID; GPU masks are obtained independently on each node through Slurm.

For each of two deterministic, perturbed 2048-atom periodic systems (LJ and
LJ/Coulomb with PPPM), the suite runs:

- A single-rank CPU reference, with GPUs still visible.
- GPU-package CUDA with one GPU, two GPUs on one node, and four on two nodes.
- Kokkos CUDA with the same three layouts, including device communication and
  CUDA-aware MPI. The PPPM case explicitly selects `pppm/kk` and cuFFT.

The 14 executions compare initial and final energy, pressure tensor and every
force component, with 20 NVE steps and a fixed energy-drift gate. One further
check validates image provenance and the executable's Git banner. Accelerator
styles are selected explicitly; missing styles cannot silently fall back.
Predeclared tolerances are `atol=rtol=2e-5` for the mixed-precision GPU package
and `2e-8` for double-precision Kokkos. NVE total-energy drift must be at most
`1e-4` per atom. The exact inputs, numerical differences and tolerances accompany
the report. These are targeted numerical integration tests, not the complete
LAMMPS unit test suite or a performance/scaling benchmark.

## Published interface

- HTML: <https://stardust0831.github.io/lammps/>
- JUnit XML: <https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/junit.xml>
- Summary JSON: <https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/summary.json>
- Raw inputs, outputs and Slurm records: <https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/raw.tar.gz>
- File hashes: <https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/manifest.json>
- Actions: <https://github.com/Stardust0831/lammps/actions/workflows/sai-multigpu.yml>

Summary fields `date`, `commit`, `branch`, `version`, `compiler`,
`operating_system` and `url` follow the upstream dashboard's external-test
interface. Additional fields describe hardware, allocation and provenance.
JUnit contains failures and infrastructure errors as well as passes. The
publisher commits XML, JSON and HTML together, including failed completed
runs. The Git banner in JUnit lets the upstream importer detect a mixed read
across report updates.

The existing upstream importer can read the public pair without source edits:

```bash
python3 tools/fetch_unittest.py --datadir /path/to/separate-test-data \
  --config sai-mpi-cuda-regression \
  --url https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/junit.xml \
  --summary-url https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/summary.json
```

For scheduled ingestion, its `SOURCES` list needs the same configuration name
and two URLs. This is an external numerical regression suite despite the
generic importer's `unit-tests` storage category. No upstream dashboard change
or upstream message is made by this fork.

## Operator setup and recovery

`site.json` holds public paths and allocation settings. New computation and
scratch live in `/org/org-open-sai/proj-sai-hpc-software/lammps-daily-tester`,
charged to `proj-sai-hpc-software`. Existing accepted images remain read-only at
their published paths. No software is installed into `/opt`.

Install the four files listed in `common.CONTROL_FILES` into the site's
`control/` directory. The VM holds a dedicated SSH key whose authorized-key
entry on SAI uses `restrict,command=".../endpoint.py"`. Only `inspect`, `start`,
`status`, `fetch` and publication `ack` commands with validated arguments are accepted. The key
cannot open a shell, transfer new executable code, forward ports or submit an
arbitrary batch script. Updating the site controller requires the operator's
separate SSH access. Each job receives a frozen copy of the installed controls.

Register the repository runner with label `sai-lammps-vm`, set `SAI_SSH_CONFIG`
in its service environment, and define host `sai-lammps-tester` in that private
configuration with pinned host keys and the dedicated identity. The workflow
has only scheduled and manual triggers, restricted to this fork; it does not
run pull-request code. A memory limit on the VM service bounds peak usage.

The hourly workflow skips unchanged upstream commits without submitting GPU
work or changing published reports. `workflow_dispatch` with `force=true`
requests a retest. One active job is resumed after a relay interruption instead
of duplicated. A new upstream commit without a matching accepted image is
reported as waiting, never as tested. Failures before any completed result
remain visible in Actions and leave the previous explicitly dated report intact.
Completed results remain resumable until the report push is acknowledged, so
a connection loss during publication does not discard a completed test.

Remote runs and failed attempts are retained. Inspect `plan.json`,
`slurm-*.log`, `accounting.json`, `results/cases.json` and `results/manifest.json`
for diagnosis. A run that lost its durable job handle is not automatically
resubmitted. A held job recorded in `plan.json` can be released by the operator.

Run the local negative tests with:

```bash
python3 -B -m unittest discover -s tools/sai -p 'test_*.py' -v
```
