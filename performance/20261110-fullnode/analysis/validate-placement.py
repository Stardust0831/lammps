#!/usr/bin/env python3
"""Check rank placement against the recorded physical CPU and GPU topology."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
result = json.loads((args.run / 'performance.json').read_text())
assert result['status'] == 'PASS'
cpu_maps = {}
for path in (args.run / 'nodes').glob('*-cpu-topology.txt'):
    rows = [line.split() for line in path.read_text().splitlines()[1:] if line.strip()]
    cpu_maps[path.name.removesuffix('-cpu-topology.txt')] = {int(cpu): (int(socket), int(core))
                                                          for cpu, core, socket, _ in rows}
validated = []
for case in result['numerical_checks'] + result['measurements']:
    name = case.get('case', case.get('name'))
    per_node = case['gpus_per_node']
    rows = [json.loads(path.read_text()) for path in (args.run / 'cases' / name / 'ranks').glob('*.json')]
    expected = case['nodes'] * per_node
    assert sorted(row['rank'] for row in rows) == list(range(expected))
    assert len({row['gpu']['uuid'] for row in rows}) == expected
    nodes = sorted({row['node'] for row in rows})
    assert len(nodes) == case['nodes']
    for node in nodes:
        local = [row for row in rows if row['node'] == node]
        assert len(local) == per_node
        assert len(set().union(*(set(row['affinity']) for row in local))) == sum(len(row['affinity']) for row in local)
        for row in local:
            assert row['cuda_visible_devices'] == row['gpu']['uuid']
            assert len({cpu_maps[node][cpu] for cpu in row['affinity']}) == 4
            assert set(row['affinity']).issubset(row['gpu']['local_cpus'])
            assert row['ucx_environment']['UCX_TLS'] == '^cuda_ipc'
    validated.append(dict(case=name, nodes=nodes, gpus_per_node=per_node, unique_gpus=expected,
                          physical_cores_per_rank=4, status='PASS'))
args.output.write_text(json.dumps(dict(status='PASS', cases=validated), indent=2) + '\n')
print('validated', len(validated), 'GPU cases, including', sum(row['unique_gpus'] == 32 for row in validated), '32-GPU cases')
