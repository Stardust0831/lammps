# Performance input provenance

`sources.json` pins every supplied data file to a source revision and SHA-256.
The operator submission copies these files into each frozen run. Each LAMMPS
working directory receives the files it needs, so the container only needs its
normal working-directory mount.

## Tungsten SNAP

`W.snapcoeff` and `W.snapparam` are the 2J8 tungsten SNAP potential from the
[UK NNSS LAMMPS benchmark](https://github.com/UKNNSS-Benchmarks/uknnss-benchmark-lammps/tree/affa74adb7ada59ae0a28fe045a5141becdb3f4c/benchmarks/common).
That repository states an MIT license for its benchmark description and files.
The potential header credits Mitchell Wood and the Wood and Thompson paper
"Quantum-Accurate Molecular Dynamics Potential for Tungsten" (2017).

The SAI input retains its BCC lattice constant (3.1803 Angstrom), atomic mass
(183.84), SNAP + ZBL interaction (Z=74, inner/outer cutoffs 4.0/4.8 Angstrom),
800 K temperature, 0.5 fs timestep and neighbor-list update policy. It uses
Nose-Hoover NVT for repeated performance measurements rather than the original
Langevin thermostat. This is an adapted scaling experiment, not a submission
under the procurement benchmark rules. The actual supplied potential is
tungsten; the source repository overview currently mentions a different model.

## SPC/E water

`data.spce` comes unchanged from LAMMPS `bench/POTENTIALS/data.spce` at the pinned
revision, under the LAMMPS GPL-2.0 license. The corresponding `in.spce` supplies
the interaction parameters, 9.8 Angstrom cutoffs, PPPM accuracy 1e-4, SHAKE
tolerance 1e-4, 300 K Nose-Hoover thermostat and 2 fs timestep.

The 4500-atom periodic cell is replicated in three directions. The initial
velocity distribution uses a fixed seed with geometry-based random numbers.
All measured pair, bonded, SHAKE, integration and PPPM styles are explicitly
Kokkos variants. The accepted build uses CUDA and cuFFT.

## Numerical checks

Before timing, the script creates a common serialized initial state on a CPU.
Both the CPU reference and each GPU layout then read that same file. This is
necessary because SHAKE corrects the original water geometry during setup.
The gate compares initial and 10-step NVE energies, temperature, pressure
tensor and every atomic force component with `atol=2e-6, rtol=2e-7`. Water
uses a 0.5 fs timestep for this short comparison. Performance runs use the
production-style thermostats and timesteps above and are not required to
retain identical long trajectories across different MPI decompositions.
