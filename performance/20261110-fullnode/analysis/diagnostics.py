#!/usr/bin/env python3
"""Describe incomplete attempts without marking a cancelled job as passed."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('evidence', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--reference-run', type=Path, required=True)
args = parser.parse_args()
root, publication = args.evidence.resolve(), args.output.resolve()
partial = root / 'topology-20261110-b'
complete = root / 'topology-noipc-20261110-a'
before = json.loads((partial / 'performance.json').read_text())
after = json.loads((complete / 'performance.json').read_text())
assert after['status'] == 'PASS'
for key in ('source_sha', 'image_sha256', 'recipe_sha256'):
    assert before['plan'][key] == after['plan'][key]
assert before['mpi'] == after['mpi']
assert before['plan'].get('runtime_environment', {}) == {}
assert after['plan']['runtime_environment'] == {'UCX_TLS': '^cuda_ipc'}
indexed = {row['name']: row for row in after['measurements']}
rows = []
rejected = []
pattern = r'\bUCX\s+ERROR\b|cuIpcOpenMemHandle\(\) failed'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranks(path):
    rows = [json.loads(p.read_text()) for p in (path / 'ranks').glob('*.json')]
    return sorted((r['rank'], r['node'], r['gpu']['uuid'], tuple(r['affinity'])) for r in rows)


for row in before['measurements']:
    current = indexed[row['name']]
    assert row['status'] == current['status'] == 'PASS'
    for key in ('kind', 'series', 'dims', 'nodes', 'gpus_per_node', 'gpus', 'steps', 'warmup', 'repeats', 'natoms'):
        assert row[key] == current[key], (row['name'], key)
    left, right = partial / 'cases' / row['name'], complete / 'cases' / row['name']
    assert ranks(left) == ranks(right), 'different physical GPU or CPU placement'
    for name in ('in.lmp', 'W.snapcoeff', 'W.snapparam') if row['kind'] == 'snap' else ('in.lmp', 'data.spce'):
        assert digest(left / name) == digest(right / name), 'changed input'
    assert 'Total wall time:' in (left / 'screen.txt').read_text()
    assert 'Total wall time:' in (right / 'screen.txt').read_text()
    assert not re.search(pattern, (right / 'screen.txt').read_text())
    case = row['kind'] + '-gate-' + str(row['gpus']) + 'gpu'
    for result in (before, after):
        assert any(g['case'] == case and g['status'] == 'PASS' for g in result['numerical_checks'])
    bad = []
    for path in (left / 'screen.txt', partial / 'cases' / case / 'screen.txt'):
        errors = [line for line in path.read_text().splitlines() if re.search(pattern, line)]
        if errors:
            bad.append(dict(path=str(path.relative_to(partial)), errors=len(errors), first_error=errors[0]))
    if bad:
        rejected.append(dict(case=row['name'], reason='UCX errors in timing or numerical-control log', logs=bad))
        continue
    a, b = statistics.median(row['loop_seconds']), statistics.median(current['loop_seconds'])
    rows.append(dict(case=row['name'], gpus=row['gpus'], default_seconds=row['loop_seconds'],
                     no_ipc_seconds=current['loop_seconds'], default_median=a, no_ipc_median=b,
                     overhead_percent=(b / a - 1) * 100))

accounting = json.loads((root / 'topology-20261110-accounting.json').read_text())
total_hours = 0
for row in accounting:
    count = int(re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', row['AllocTRES'])[1])
    row['allocated_gpu_hours'] = count * int(row['ElapsedRaw']) / 3600
    total_hours += row['allocated_gpu_hours']

reference = json.loads((args.reference_run / 'performance.json').read_text())
for key in ('source_sha', 'image_sha256', 'recipe_sha256'):
    assert reference['plan'][key] == after['plan'][key]
assert reference['mpi'] == after['mpi']
reference_rows = {row['name']: row for row in reference['measurements']}
for row in after['measurements']:
    old = reference_rows[row['name']]
    for key in ('kind', 'series', 'dims', 'steps', 'warmup', 'repeats', 'natoms', 'gpus'):
        assert old[key] == row[key]
    for filename in ('in.lmp', 'W.snapcoeff', 'W.snapparam') if row['kind'] == 'snap' else ('in.lmp', 'data.spce'):
        assert digest(complete / 'cases' / row['name'] / filename) == digest(args.reference_run / 'cases' / row['name'] / filename)
reference_errors = []
for path in sorted((args.reference_run / 'cases').glob('*/screen.txt')):
    errors = [line for line in path.read_text().splitlines() if re.search(pattern, line)]
    if errors:
        reference_errors.append(dict(path=str(path.relative_to(args.reference_run)), errors=len(errors), first_error=errors[0]))
assert reference_errors, 'the historical correction requires raw evidence'

scope = ('The default-transport full-node job was cancelled after water/PPPM became extremely slow. '
         'A full log audit found UCX errors in eight of its 12 completed timings and in all four '
         'water numerical gates. It is not a passed full experiment. This table uses only its '
         'four completed, error-free SNAP points '
         'whose CPU/GPU numerical checks passed. The paired run uses the same image, inputs, '
         'physical GPU UUIDs and CPU affinity, but excludes CUDA IPC. Measurement order and time '
         'differ, so this is diagnostic evidence, not a randomized causal trial.')
record = dict(scope=scope, default_job=before['plan']['job'], no_ipc_job=after['plan']['job'],
              comparisons=rows, rejected_comparisons=rejected,
              historical_reference=dict(job=reference['plan']['job'], original_status=reference['status'],
                                        revised_acceptance='WITHDRAWN', errors=reference_errors,
                                        matched_input_decks=len(after['measurements'])),
              attempts=accounting, allocated_gpu_hours=total_hours)
(publication / 'diagnostics.json').write_text(json.dumps(record, indent=2) + '\n')
lines = ['\n## Communication workaround and retained attempts\n', scope, '',
         '| Completed point | Default median s | No CUDA IPC median s | Change |',
         '|---|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['case']} | {row['default_median']:.3f} | {row['no_ipc_median']:.3f} | {row['overhead_percent']:+.1f}% |")
lines += ['', 'No reliable water overhead can be calculated from the default-transport timings: '
          'all completed default water points contain communication errors. Disabling the IPC mapping cache and forcing whole '
          'allocation registration each reproduced UCX errors in isolated probes. Excluding '
          'the CUDA IPC transport completed the isolated 16-GPU water test and the final full matrix.', '',
          'No SIF, application binary, system MPI installation or public module was changed. '
          'The workaround is an explicit job environment setting; GPU calculation remains enabled.', '',
          'The earlier [four-GPU-per-node report](../20261110/README.md) from job `2338108` also '
          'contains UCX errors in 13 of 28 case logs, including four numerical gates. Its '
          'original success verdict is withdrawn. The input decks, models, image, source and '
          'host MPI match this rerun, but it is not an accepted scaling baseline. This report '
          'does not attribute the difference between those runs to node packing or quantify '
          'the isolated cost of disabling CUDA IPC.', '',
          'The tested nodes contain four separate four-GPU NVLink groups. The launcher selects '
          'GPUs by CPU locality rather than assuming full-node GPU indices follow CPU core order. '
          'Every GPU rank has four physical CPU cores with disjoint affinity and a unique GPU UUID; '
          'the two-node cases use all 32 physical GPUs. See [placement validation](placement-validation.json).', '',
          f'The six performance and communication attempts listed below used **{total_hours:.2f} allocated GPU-hours**, '
          'including cancelled runs, failed probes and reserved GPUs during smaller layouts.', '',
          '| Job | Run | State | Elapsed | Allocated GPU-hours |', '|---|---|---|---:|---:|']
for row in accounting:
    name = row['WorkDir'].rsplit('/', 1)[-1]
    link = '[raw](raw.tar.gz)' if name == after['plan']['run_id'] else f'[raw](attempts/{name}.tar.gz)'
    lines.append(f"| {row['JobIDRaw']} | {name} ({link}) | {row['State']} | {row['Elapsed']} | {row['allocated_gpu_hours']:.2f} |")
lines += ['', '[Diagnostic data and accounting](diagnostics.json). The raw archives retain '
          'failed/cancelled status and operator explanations. [UCX PR #11216]'
          '(https://github.com/openucx/ucx/pull/11216) addresses the same error class; '
          'its patch was not applied or tested here.', '']
with (publication / 'README.md').open('a') as stream:
    stream.write('\n'.join(lines))
print(json.dumps(dict(completed_pairs=len(rows), allocated_gpu_hours=total_hours)))
