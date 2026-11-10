#!/usr/bin/env python3
"""Reject incomplete timings and silently changed numerical evidence."""
import unittest

from performance import input_text, measured_loops, states


class PerformanceGates(unittest.TestCase):
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
