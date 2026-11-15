#!/usr/bin/env python3
"""Negative checks for numerical gates, report integrity and duplicate jobs."""
import io
import math
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, call, patch
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import endpoint
import relay
import suite


class TesterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sha = "a" * 40
        self.accepted = dict(source_sha=self.sha, image_sha256="d" * 64,
                             recipe_sha256="e" * 64, control_digest="c" * 64)

    def test_numerical_gate_rejects_error_and_nan(self):
        for actual in ([.01], [math.nan], [math.inf], []):
            with self.subTest(actual=actual), self.assertRaises(AssertionError):
                suite.compare([0.0], actual, atol=2e-8, rtol=2e-8)
        self.assertLess(suite.compare([1.0], [1 + 1e-10], atol=2e-8, rtol=2e-8), 2e-8)

    def test_missing_thermo_is_an_error(self):
        path = self.root / "screen"
        path.write_text("LAMMPS ran, but no completed output\n")
        with self.assertRaises(ValueError):
            suite.values(path)

    def test_nonzero_exit_is_never_a_pass(self):
        with self.assertRaises(RuntimeError):
            suite.execute([sys.executable, "-c", "print('looks successful'); raise SystemExit(3)"],
                          self.root / "log")

    def test_timeout_terminates_process(self):
        with self.assertRaises(RuntimeError):
            suite.execute([sys.executable, "-c", "import time; time.sleep(20)"],
                          self.root / "log", timeout=.1)

    def test_junit_preserves_failures_and_metadata(self):
        common.write(self.root / "plan.json", dict(source_sha=self.sha, report_url="https://example.org/",
                     action_url="https://github.com/example/run/1", controller_sha="b" * 40,
                     run_id="github-1-1", control_digest="c" * 64, image_sha256="d" * 64,
                     recipe_sha256="e" * 64, job="123"))
        result = common.reports(self.root, [dict(name="numerical", status="failure", message="force mismatch"),
                                          dict(name="transport", status="error", message="MPI failed")])
        self.assertEqual(result["status"], "FAIL")
        xml = ET.parse(self.root / "results/junit.xml").getroot()
        self.assertEqual(xml.attrib["tests"], "2")
        self.assertEqual(len(xml.findall("testcase/failure")), 1)
        self.assertEqual(len(xml.findall("testcase/error")), 1)
        self.assertEqual(result["commit"], self.sha)

    def test_unchanged_accepted_build_skips_submission(self):
        common.write(self.root / "state.json", dict(last=dict(self.accepted, status="PASS")))
        with patch.object(endpoint, "snapshot", return_value=self.accepted), \
                patch.object(endpoint.subprocess, "run") as command:
            result = endpoint.start(dict(root=str(self.root)), "github-2-1", self.sha, "b" * 40, "once")
        self.assertTrue(result["skipped"])
        command.assert_not_called()

    def test_rebuilt_image_or_changed_tests_are_submitted_for_the_same_source(self):
        for index, key in enumerate(("image_sha256", "recipe_sha256", "control_digest", "legacy", "force"), 1):
            with self.subTest(key=key):
                last = dict(self.accepted, status="PASS")
                if key == "legacy":
                    last = dict(source_sha=self.sha, status="PASS")
                elif key != "force":
                    last[key] = "f" * 64
                common.write(self.root / "state.json", dict(last=last))
                run_id = f"github-2-{index}"
                with patch.object(endpoint, "snapshot", return_value=self.accepted), \
                        patch.object(endpoint, "batch", return_value="#!/bin/bash\nexit 0\n"), \
                        patch.object(endpoint.subprocess, "run", return_value=Mock(
                            returncode=0, stdout="123\n", stderr="")) as command:
                    result = endpoint.start(dict(root=str(self.root), repository="example/lammps",
                                            report_url="https://example.org/", partition="16V100"),
                                            run_id, self.sha, "b" * 40, "force" if key == "force" else "once")
                self.assertEqual(result["state"], "SUBMITTED")
                plan = common.read(self.root / "runs" / run_id / "plan.json")
                self.assertEqual(plan["source_sha"], self.sha)
                self.assertEqual(plan["image_sha256"], self.accepted["image_sha256"])
                self.assertEqual(command.call_args_list[1].args[0][:3], ["sbatch", "--hold", "--parsable"])

    def test_completed_result_retains_build_identity_for_duplicate_suppression(self):
        plan = dict(self.accepted, run_id="github-1-1", job="123", controller_sha="b" * 40,
                    report_url="https://example.org/", action_url="https://example.org/run/1")
        common.write(self.root / "plan.json", plan)
        common.reports(self.root, [dict(name="numerical", status="pass", message="")])
        config = dict(root=str(self.root), partition="16V100", account="project")
        row = ["123", "COMPLETED", "0:0", "16V100", "project", str(self.root)]
        with patch.object(endpoint, "accounting", return_value=row):
            final = endpoint.status(self.root, config)
        for key, value in self.accepted.items():
            self.assertEqual(final[key], value)
        common.write(self.root / "state.json", dict(last=final))
        with patch.object(endpoint, "snapshot", return_value=self.accepted), \
                patch.object(endpoint.subprocess, "run") as command:
            self.assertTrue(endpoint.start(config, "github-2-1", self.sha, "b" * 40, "once")["skipped"])
        command.assert_not_called()

    def test_active_job_is_resumed_even_when_head_changes(self):
        common.write(self.root / "state.json", dict(active="github-1-1"))
        with patch.object(endpoint, "status", return_value=dict(terminal=False, job="123")), \
                patch.object(endpoint.subprocess, "check_output") as command:
            result = endpoint.start(dict(root=str(self.root)), "github-2-1", self.sha, "b" * 40, "once")
        self.assertTrue(result["resumed"])
        self.assertEqual(result["job"], "123")
        command.assert_not_called()

    def test_rotated_build_does_not_claim_the_requested_source_was_tested(self):
        common.write(self.root / "state.json", dict(last=dict(self.accepted, status="PASS")))
        with patch.object(endpoint, "snapshot", return_value=dict(source_sha="b" * 40)), \
                patch.object(endpoint.subprocess, "run") as command:
            result = endpoint.start(dict(root=str(self.root)), "github-2-1", self.sha, "b" * 40, "once")
        self.assertTrue(result["pending"])
        command.assert_not_called()

    def test_relay_tests_accepted_build_without_querying_upstream_head(self):
        expected = dict(terminal=False, job="123", source_sha=self.sha)
        with patch.object(relay, "remote", side_effect=[self.accepted, expected]) as remote, \
                patch.object(relay, "control_digest", return_value=self.accepted["control_digest"]), \
                patch.object(relay.subprocess, "check_output", side_effect=AssertionError("no upstream query")):
            self.assertEqual(relay.start_accepted("github-2-1", "b" * 40, "once"), expected)
        self.assertEqual(remote.call_args_list, [call("inspect"),
                         call("start", "github-2-1", self.sha, "b" * 40, "once")])

    def test_relay_refreshes_a_build_that_rotates_before_submission(self):
        newer = dict(self.accepted, source_sha="f" * 40, image_sha256="f" * 64)
        expected = dict(terminal=False, job="123", source_sha=newer["source_sha"])
        with patch.object(relay, "remote", side_effect=[self.accepted, dict(pending=True), newer, expected]) as remote, \
                patch.object(relay, "control_digest", return_value=self.accepted["control_digest"]):
            self.assertEqual(relay.start_accepted("github-2-1", "b" * 40, "once"), expected)
        self.assertEqual(remote.call_args_list[-1], call("start", "github-2-1", newer["source_sha"], "b" * 40, "once"))

    def test_repeated_build_rotation_fails_instead_of_reporting_success(self):
        with patch.object(relay, "remote", side_effect=[self.accepted, dict(pending=True)] * 3), \
                patch.object(relay, "control_digest", return_value=self.accepted["control_digest"]), \
                self.assertRaisesRegex(RuntimeError, "changed repeatedly"):
            relay.start_accepted("github-2-1", "b" * 40, "once")

    def test_controller_mismatch_cannot_submit_an_accepted_build(self):
        with patch.object(relay, "remote", return_value=self.accepted) as remote, \
                patch.object(relay, "control_digest", return_value="f" * 64), \
                self.assertRaisesRegex(ValueError, "controller differs"):
            relay.start_accepted("github-2-1", "b" * 40, "once")
        remote.assert_called_once_with("inspect")

    def test_completed_unpublished_job_is_resumed(self):
        common.write(self.root / "state.json", dict(active="github-1-1"))
        with patch.object(endpoint, "status", return_value=dict(terminal=True, job="123", status="PASS")), \
                patch.object(endpoint.subprocess, "check_output") as command:
            result = endpoint.start(dict(root=str(self.root)), "github-2-1", self.sha, "b" * 40, "once")
        self.assertTrue(result["resumed"])
        self.assertTrue(result["terminal"])
        self.assertEqual(common.read(self.root / "state.json")["active"], "github-1-1")
        command.assert_not_called()

    def test_acknowledgement_is_required_before_skipping(self):
        common.write(self.root / "site.json", dict(root=str(self.root)))
        common.write(self.root / "state.json", dict(active="github-1-1"))
        final = dict(terminal=True, job="123", run_id="github-1-1", source_sha=self.sha, status="PASS")
        with patch.object(endpoint, "CONTROL", self.root), patch.object(endpoint, "status", return_value=final):
            for _ in range(2):
                self.assertTrue(endpoint.main(["ack", "github-1-1"])["acknowledged"])
        state = common.read(self.root / "state.json")
        self.assertNotIn("active", state)
        self.assertEqual(state["last"], final)

    def test_restricted_endpoint_rejects_injected_commands(self):
        for name in ("../../escape", "github-1-1; touch /tmp/example", "$(id)"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                endpoint.start({}, name, self.sha, self.sha, "once")
        with self.assertRaises(ValueError):
            endpoint.main(["bash", "-c", "id"])

    def test_report_archive_cannot_escape_destination(self):
        raw = io.BytesIO()
        with tarfile.open(fileobj=raw, mode="w:gz") as archive:
            record = tarfile.TarInfo("../escaped")
            record.size = 1
            archive.addfile(record, io.BytesIO(b"x"))
        with self.assertRaises(ValueError):
            relay.collect(raw.getvalue(), self.root / "artifacts", self.sha)
        self.assertFalse((self.root / "escaped").exists())


if __name__ == "__main__":
    unittest.main()
