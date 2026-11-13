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
The executable reports `-modified` for that integration. Reports retain the
verbatim binary banner and flag this explicitly; this is a test of the site's
accepted build of the recorded upstream revision.

## Tests

One job requests two nodes with four V100 GPUs per node in the `16V100`
partition. Each physical node has 16 GPUs; the small regression suite uses a
subset to exercise cross-node communication. Host Open MPI launches the accepted executable through its existing
Apptainer wrapper. Every rank records its node, CPU affinity and allocated GPU
UUID; GPU masks are obtained independently on each node through Slurm.

For each of two deterministic, perturbed 2048-atom periodic systems (LJ and
LJ/Coulomb with PPPM), the suite runs:

- A single-rank CPU reference, with GPUs still visible.
- GPU-package CUDA with one, two and four GPUs on one node, and eight on two nodes,
  explicitly using `pppm/gpu` for the charged fixture.
- Kokkos CUDA with the same four layouts, including device communication and
  CUDA-aware MPI. The PPPM case explicitly selects `pppm/kk` and cuFFT.

The 18 executions compare initial and final energy, pressure tensor and every
force component, with 20 NVE steps and a fixed energy-drift gate. One further
check validates image provenance and the executable's Git banner. Accelerator
styles are selected explicitly; missing styles cannot silently fall back.
Predeclared tolerances are `atol=rtol=2e-5` for the mixed-precision GPU package
and `2e-8` for double-precision Kokkos. NVE total-energy drift must be at most
`1e-4` per atom. The exact inputs, numerical differences and tolerances accompany
the report. These are targeted numerical integration tests, not the complete
LAMMPS unit test suite or a performance/scaling benchmark.

## Verified example

The [clean rerun](https://github.com/Stardust0831/lammps/actions/runs/42760147255)
completed the full runner-to-Slurm-to-publication path with 19/19 checks
passing. Slurm job `2341678` ran on two nodes in 71 seconds with
`UCX_TLS=^cuda_ipc`. All 18 case logs are free of UCX errors; all 63 rank
records carry the declared transport setting. Its
[immutable report](https://github.com/Stardust0831/lammps/tree/5f02b22e47ad64cc5d2388ffd81274addf2f91a1)
retains 209 files of hashed raw evidence. This rerun used the newer accepted
daily image of the same upstream commit; performance experiments keep their
separate image pin.

The [first completed end-to-end run](https://github.com/Stardust0831/lammps/actions/runs/42671424209/attempts/2)
tested upstream commit `d291d572cf3c73a8a720774b78482a7c931304ca`. Slurm job
`2332512` used eight V100 GPUs and exited `0:0`; all 19 numerical/provenance
checks originally passed. A subsequent raw-log audit found CUDA IPC errors
in six Kokkos cases, so that original success verdict is withdrawn. An
[immutable copy of that report](https://github.com/Stardust0831/lammps/tree/531f9018150c14b35d80175a10e7e30fe2ba1a36)
includes 209 files of unchanged raw evidence. Numerical agreement and a zero
exit code alone do not make a run with communication errors acceptable.

The unmodified `fetch_unittest.py` and `generator/build_site.py` from
`lammps/lammps-test-results` commit `954defea44c3b90ed0a30e6899d6e975bf5574f1`
successfully imported the public URLs below and generated dashboard pages
with the original 19-pass verdict. This verified the import interface, not
the now-withdrawn execution acceptance. A second import
archived no additional run. A separate
[unchanged-commit workflow run](https://github.com/Stardust0831/lammps/actions/runs/42672966585)
submitted no GPU job and left the published report commit unchanged.

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
runs. GitHub Pages is deployed explicitly through Actions. JUnit records a
normalized upstream ref as `upstream_source_git_info`, followed by the actual
`binary_git_info`. The upstream importer can parse the full source SHA from
the former and detect a mixed read across report updates, even when the
binary's detached-HEAD banner contains a bare hash with `-modified`.

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
Submissions explicitly select the existing project group with `sg` before
calling `sbatch`. The plan records the caller's group IDs and the selected
submission group, so filesystem access at job startup can be audited.

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

The VM's hourly timer requests the workflow, which skips unchanged upstream commits without submitting GPU
work or changing published reports. `workflow_dispatch` with `force=true`
requests a retest. One active job is resumed after a relay interruption instead
of duplicated. A new upstream commit without a matching accepted image is
reported as waiting, never as tested. Failures before any completed result
remain visible in Actions and leave the previous explicitly dated report intact.
Completed results remain resumable until the report push is acknowledged, so
a connection loss during publication does not discard a completed test.
Acknowledgement happens after both the report-branch push and Pages deployment.

### Local trigger

`scheduler/lammps-sai-scheduler.timer` triggers at minute 17 of each hour in
`Asia/Shanghai`. The existing workflow remains available for manual dispatch;
GitHub cron is disabled. A startup check and `Persistent=true` recover a missed
tick when the user manager starts. The VM and its WSL distribution must be
running; the timer cannot wake a powered-off Windows host.

The short-lived Python dispatcher only calls the GitHub API. It skips requests
while a workflow is queued or running, locks concurrent local invocations, and
records its intent before sending the request. A lost HTTP reply imposes a
15-minute delay before retrying; GitHub workflow concurrency and the existing
cluster state prevent duplicate GPU jobs. Every request uses `force=false`.
API failures remain visible in the service journal and retry at the next tick.

Install under a trusted operator account with user lingering enabled, outside
the runner's job directory. The deployed VM uses the existing `sai` scheduler
account, separate from the LAMMPS runner account. The dispatcher needs an
owner-only token file with repository Actions read/write permission, normally
`~/.config/lammps-sai-scheduler/token`. An existing credential can be selected
by putting `LAMMPS_SAI_TOKEN_FILE=/absolute/path/to/token` in
`~/.config/lammps-sai-scheduler/environment`; no token is stored in the workflow.

```bash
bash tools/sai/scheduler/install.sh --install
systemctl --user list-timers lammps-sai-scheduler.timer
journalctl --user -u lammps-sai-scheduler.service -n 30
```

`--prepare` copies the files without enabling the timer. The oneshot service
has a 64 MiB memory limit and no swap; no Python dispatcher stays resident.
The existing self-hosted runner still executes the workflow and SAI still
performs the scientific calculations. A successful dispatch only establishes
that GitHub accepted the request; inspect Actions and the dated public report
to verify execution.

The site configuration now records `UCX_TLS=^cuda_ipc` for the small scheduled
suite. It disables the problematic CUDA IPC transport while retaining GPU
calculation. Reports and per-rank records disclose this setting. All test
launches fail on UCX errors even if LAMMPS later returns zero. This setting
does not modify public modules, MPI installations or the accepted image.

Remote runs and failed attempts are retained. Inspect `plan.json`,
`slurm-*.log`, `accounting.json`, `results/cases.json` and `results/manifest.json`
for diagnosis. A run that lost its durable job handle is not automatically
resubmitted. A held job recorded in `plan.json` can be released by the operator.

Run the local negative tests with:

```bash
python3 -B -m unittest discover -s tools/sai -p 'test_*.py' -v
```

## Multi-node performance experiments

`submit_performance.py` provides an operator-triggered scaling experiment on
the same accepted image. It is separate from the hourly regression trigger.
The [input provenance](benchmarks/README.md) describes the tungsten SNAP and
SPC/E water models and their numerical controls.

Run a two-node pilot from a checkout on SAI:

```bash
python3 -B tools/sai/submit_performance.py \
  /org/org-open-sai/proj-sai-hpc-software/lammps-performance/runs/pilot-UNIQUE \
  --nodes 2 --profile pilot
```

Once that run passes, use its timings to choose the measurement length:

```bash
python3 -B tools/sai/submit_performance.py \
  /org/org-open-sai/proj-sai-hpc-software/lammps-performance/runs/scaling-UNIQUE \
  --nodes 8 --profile scaling \
  --calibration /org/org-open-sai/proj-sai-hpc-software/lammps-performance/runs/pilot-UNIQUE/performance.json
```

The script freezes its controller, input hashes and module generation before
submitting a held job, records the job number, then releases it. Existing run
directories are never overwritten. A 32-GPU run uses the project's permitted
`huge-gpu` QoS because `rush-gpu` allows at most 16 GPUs per user/job. All runs
have a 45-minute limit, and an individual process has a 20-minute limit.

The scaling matrix uses four GPUs per node and 1/2/4/8 nodes. Strong scaling
keeps 2,097,152 W atoms or 972,000 water atoms fixed. Weak scaling starts from
524,288 W atoms or 121,500 water atoms on four GPUs, then increases the atom
count with the number of GPUs. Each point has 100 warmup steps and three
timed segments, with fixed steps within a series. CPU/GPU numerical checks
precede the large runs at every layout. Per-node GPU telemetry and every
rank's GPU UUID and CPU binding accompany the raw logs.

The launcher uses host MPI with `ppr:4:node:pe=4`, matching the site's
four physical cores per GPU, and one OpenMP thread per rank for Kokkos CUDA.
The initial pilot used a single physical core per rank; its purpose is sizing
and calibration. Scaling comparisons use the same final mapping throughout.
The jobs request four GPUs on each 16-GPU node without exclusive-node access,
so the results can contain interference from other jobs.

To compare that layout with full nodes, reuse the completed scaling experiment:

```bash
python3 -B tools/sai/submit_performance.py \
  /org/org-open-sai/proj-sai-hpc-software/lammps-performance/runs/topology-UNIQUE \
  --nodes 2 --profile topology \
  --reference /org/org-open-sai/proj-sai-hpc-software/lammps-performance/runs/scaling-UNIQUE/performance.json
```

This requests two nodes with 16 GPUs each. The measurement layouts are 1x4,
1x8, 1x16 and 2x16 (nodes x GPUs per node). It pins the reference image and
module, copies the same sizes, timesteps, warmup and sample counts, and freezes
the reference record. Every layout receives CPU/GPU force and thermo checks.
The rank count, device masks, CPU binding, GPU topology and actual allocation
are recorded. In this mode, GPUs are ordered by their PCI CPU-locality groups
to match sequential MPI core binding. Every rank checks that its bound CPUs
are local to its selected GPU before starting LAMMPS. This matters when GPU
indices run in the opposite order to CPU NUMA domains. The allocation retains
all 32 GPUs while smaller cases run;
GPU-hour accounting includes that idle capacity. This manual mode does not
change the small scheduled regression suite.

For a targeted diagnosis, `--only spce-strong-16gpu` limits performance work
while retaining the numerical checks. `--ucx-cuda-ipc-cache n` explicitly
disables UCX's CUDA IPC mapping cache for that experiment. The plan and rank
records retain this setting, and the runtime forwards it into the SIF. This
is an experimental override, not a site-wide default. Comparisons must state
when their communication settings differ. UCX errors terminate an experiment
promptly and cannot be accepted as successful timings.

Other diagnostic options are `--ucx-whole-allocation on` and
`--disable-cuda-ipc` (sets `UCX_TLS=^cuda_ipc`). The latter retains GPU
calculation but changes the communication transport and can reduce throughput.
`--first` lists points to run first without dropping the rest; `--qos` selects
one of the project's two supported GPU queues. In the 2026-11-10 full-node
trial, UCX 1.20.0 reported `cuIpcOpenMemHandle(): resource already mapped`
during PPPM reinitialization. Cache-off and whole-allocation registration
did not resolve the error. Disabling CUDA IPC passed the isolated 16-GPU
water test and the complete 1x4/1x8/1x16/2x16 matrix. Upstream
[UCX PR #11216](https://github.com/openucx/ucx/pull/11216)
addresses the same error class and was still open when checked. This does not
establish that the proposed upstream patch fixes this particular workload.

After copying a completed run and its Slurm accounting to an analysis host,
`report_performance.py` validates the completion record, frozen inputs and
every fixture, numerical-gate and performance log for clean exits and UCX errors,
then writes CSV, Markdown, PNG/SVG curves and the upstream performance JSON:

```bash
python3 -B tools/sai/report_performance.py /path/to/copied/run \
  --accounting /path/to/accounting.json --output /path/to/report
```

For a topology run, add `--reference-run /path/to/copied/scaling-run` to produce
comparison tables and plots. It checks identical image bytes, CPU/GPU models,
GPU driver, actual input decks and model/data files before comparing timings.
Use the intended publication directory name for `--output`, because its last
component becomes the public report URL. Separate allocations can differ in
placement and contention; the repeated 1x4 point helps expose that variation.

Only the report generator needs Matplotlib. The cluster and VM controllers
use Python's standard library. The report uses median timings with sample
ranges; the upstream dashboard currently chooses the best sample from the
same published measurements. Large-run finite-state checks and short CPU
force comparisons do not establish a converged production trajectory.
The raw timing sections are retained for diagnosis, but asynchronous CUDA
execution can charge GPU waits to `Comm` or another later section. Follow
the Kokkos guide's `CUDA_LAUNCH_BLOCKING=1` advice in a separate diagnostic
run before attributing those sections to a specific bottleneck.

The [2026-11-10 full-node trial](https://github.com/Stardust0831/lammps/tree/sai-multigpu-reports/performance/20261110-fullnode)
completed all 16 performance points and eight CPU/GPU numerical checks.
Slurm job `2340456` allocated two nodes with 16 V100 GPUs each and exited
`0:0` in 29m30s. All 28 case logs are free of UCX errors with CUDA IPC disabled.
The earlier four-GPU-per-node job `2338108` had communication errors despite
its original success flag; it is retained as diagnostic evidence, not an
accepted baseline. Neither its timing ratios nor cancelled default-transport
attempts establish the isolated performance cost of this workaround.

The unmodified upstream performance importer can read the public result:

```bash
python3 tools/fetch_performance.py --datadir /path/to/separate-test-data \
  --machine sai-fullnode-noipc \
  --url https://raw.githubusercontent.com/Stardust0831/lammps/sai-multigpu-reports/performance/20261110-fullnode/performance-summary.json
```

The optional reporter argument `--build-summary` supplies compiler metadata
from an earlier run only if both the image SHA-256 and source commit match.
The supplied record and its origin are retained with the performance report.
