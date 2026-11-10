#!/usr/bin/env python3
"""Reject incomplete timings and silently changed numerical evidence."""
import unittest

from performance import cpu_list, input_text, layout_sizes, local_device_order, measured_loops, states
from submit_performance import topology_benchmarks


class PerformanceGates(unittest.TestCase):
    def test_gpu_numbering_does_not_determine_cpu_locality(self):
        devices = [dict(index=str(i), uuid="gpu-" + str(i)) for i in range(16)]
        locality = {str(i): f"{(3 - i // 4) * 16}-{(4 - i // 4) * 16 - 1}" for i in range(16)}
        ordered = local_device_order(devices, locality)
        self.assertEqual([d["index"] for d in ordered],
                         list(map(str, [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3])))
        self.assertEqual(len({d["uuid"] for d in ordered}), 16)
        for rank, device in enumerate(ordered):
            self.assertTrue(set(range(rank * 4, rank * 4 + 4)).issubset(device["local_cpus"]))
        self.assertEqual(cpu_list("0-3,64-67\n"), [0, 1, 2, 3, 64, 65, 66, 67])
        for bad in ("", "3-1", "0-three"):
            with self.assertRaises(ValueError):
                cpu_list(bad)

    def test_full_nodes_preserve_experiment(self):
        rows = [dict(name=f"{kind}-{series}-{nodes * 4}gpu", kind=kind, series=series,
                     nodes=nodes, dims=[64, 64, 64], steps=1250, warmup=100, repeats=3)
                for kind in ("snap", "spce") for series in ("strong", "weak") for nodes in (1, 2, 4, 8)]
        reference = dict(status="PASS", plan=dict(profile="scaling", benchmarks=rows))
        layouts, updated = topology_benchmarks(reference, 2)
        self.assertEqual([layout_sizes(r) for r in layouts], [(1, 4), (1, 8), (1, 16), (2, 16)])
        for before, after in zip(rows, updated):
            self.assertEqual(before["nodes"] * 4, after["nodes"] * after["gpus_per_node"])
            self.assertEqual({k: v for k, v in before.items() if k != "nodes"},
                             {k: v for k, v in after.items() if k not in ("nodes", "gpus_per_node")})
        self.assertEqual(rows[-1]["nodes"], 8)
        self.assertEqual(layout_sizes(8), (8, 4))
        for bad in (dict(status="FAIL", plan=reference["plan"]),
                    dict(status="PASS", plan=dict(profile="scaling", benchmarks=rows[:-1]))):
            with self.assertRaises(ValueError):
                topology_benchmarks(bad, 2)
        with self.assertRaises(ValueError):
            topology_benchmarks(reference, 8)

    def test_warmup_is_not_a_sample(self):
        text = ("Loop time of 10 on 4 procs for 100 steps with 123 atoms\n"
                "BENCH_SAMPLE 0\nLoop time of 2 on 4 procs for 50 steps with 123 atoms\n"
                "BENCH_SAMPLE 1\nLoop time of 3 on 4 procs for 50 steps with 123 atoms\n")
        self.assertEqual(measured_loops(text, 50, 2, 123), [2, 3])
        for wrong in (text.replace("123 atoms", "124 atoms"), text.replace("50 steps", "49 steps"),
                      text.replace("BENCH_SAMPLE 1", "MISSING"), text.replace("time of 2", "time of 0")):
            with self.assertRaises(ValueError):
                measured_loops(wrong, 50, 2, 123)

    def test_missing_and_nonfinite_states_fail(self):
        line = "BENCH_STATE initial 100 " + " ".join(["1"] * 10) + "\n"
        text = line + line.replace("initial", "final")
        self.assertEqual(states(text)["initial"]["natoms"], 100)
        for wrong in (line, text + line, text.replace("final 100", "final 99"), text.replace("1 ", "nan ", 1)):
            with self.assertRaises(ValueError):
                states(wrong)

    def test_accelerated_physics_are_explicit(self):
        water = input_text("spce", (3, 3, 3), "kokkos")
        for style in ("lj/cut/coul/long/kk", "pppm/kk", "shake/kk", "harmonic/kk", "nvt/kk"):
            self.assertIn(style, water)
        snap = input_text("snap", (64, 64, 64), "kokkos")
        self.assertIn("snap/kk", snap)
        self.assertIn("zbl/kk", snap)


if __name__ == "__main__":
    unittest.main()
