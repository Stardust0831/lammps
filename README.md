# SAI LAMMPS multi-GPU tests

[Readable report](https://stardust0831.github.io/lammps/) | [Summary](summary.json) | [JUnit](junit.xml) | [Raw evidence](raw.tar.gz) | [Hashes](manifest.json)

LJ and PPPM comparisons using the GPU package and Kokkos CUDA on one and two nodes. The source commit identifies the upstream develop code in the tested accepted image. Unchanged upstream commits are skipped unless a retest is requested. CPU execution supplies numerical controls; timing is diagnostic, not a scaling claim.

[Workflow and operator documentation](https://github.com/Stardust0831/lammps/tree/develop/tools/sai)
