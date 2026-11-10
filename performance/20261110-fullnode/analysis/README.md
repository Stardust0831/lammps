# Reproducing this report

The final performance job used controller commit
`1f87db105ba812ac4bbc1fda485913a25cfdf48b`. Its exact files and hashes are in
the raw archive and `plan.json`. Reporting and the follow-up regression
controller use `4d61cd58d16d2012cf2172c0bcc3f9e0dec092e1` from
`Stardust0831/lammps`. The application source and accepted image were not rebuilt.

Unpack `../raw.tar.gz` and each archive under `../attempts/` into a separate
evidence directory. Preserve their run-name directory prefixes. Obtain the
original `scaling-20261110-b` input record from the raw archive in
`../../20261110/`. That original run is used only to verify identical inputs
and document the withdrawn acceptance; it is not a clean timing reference.
The manifests list the archive checksum and every included file checksum.
`archive-validation.json` records verification of all six follow-up archives.

With Matplotlib available, run the copied reporter against the final run:

```bash
python3 -B report_performance.py /path/to/evidence/topology-noipc-20261110-a \
  --accounting /path/to/evidence/topology-noipc-20261110-a/accounting.json \
  --output /path/to/report/20261110-fullnode \
  --build-summary ../build-summary.json
python3 -B validate-placement.py /path/to/evidence/topology-noipc-20261110-a \
  /path/to/report/20261110-fullnode/placement-validation.json
```

Do not supply the original run as `--reference-run`: the reporter now rejects
its communication errors. `diagnostics.py` uses the retained partial attempts
and the `topology-20261110-accounting.json` export to append the diagnostic
table and attempt accounting. That export is reproduced in the `attempts`
field of `../diagnostics.json`.

The runtime environment snapshots were taken while actual container processes
were running, including 16 distinct GPU masks on each of two nodes.
The numerical fixtures compare every force plus energy and pressure;
large-system validation and the placement audit complement those checks.
The 19 local tests and required `make -C src check` cover the final tester
changes. They are separate from the eight numerical gates and 16 measured
performance points in Slurm job `2340456`.
