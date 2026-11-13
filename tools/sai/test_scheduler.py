#!/usr/bin/env python3
"""Check duplicate suppression, lost responses and dispatch credentials."""
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduler import dispatch


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.now = datetime(2026, 11, 13, 4, 17, tzinfo=timezone.utc)
        self.github = Mock()
        self.github.runs.return_value = []

    def test_active_or_queued_job_prevents_another_dispatch(self):
        for status in ("queued", "in_progress", "waiting", "requested", "pending"):
            with self.subTest(status=status):
                self.github.runs.return_value = [{"status": status}]
                self.assertEqual(dispatch.tick(self.root, self.github, self.now), "busy")
        self.github.dispatch.assert_not_called()

    def test_lost_response_waits_then_checks_existing_run(self):
        self.github.dispatch.side_effect = TimeoutError("lost response")
        with self.assertRaises(TimeoutError):
            dispatch.tick(self.root, self.github, self.now)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertFalse(receipt["accepted"])
        self.assertEqual(dispatch.tick(self.root, self.github, self.now + timedelta(minutes=5)),
                         "awaiting-dispatch-visibility")
        self.github.runs.return_value = [{"status": "in_progress"}]
        self.assertEqual(dispatch.tick(self.root, self.github, self.now + timedelta(minutes=20)),
                         "busy")
        self.github.dispatch.assert_called_once()

    def test_uncertain_dispatch_can_retry_after_visibility_grace(self):
        self.github.dispatch.side_effect = [TimeoutError("lost response"), None]
        with self.assertRaises(TimeoutError):
            dispatch.tick(self.root, self.github, self.now)
        self.assertEqual(dispatch.tick(self.root, self.github, self.now + timedelta(hours=1)),
                         "dispatched")
        self.assertTrue(json.loads((self.root / "receipt.json").read_text())["accepted"])

    def test_success_is_not_dispatched_again_immediately(self):
        self.assertEqual(dispatch.tick(self.root, self.github, self.now), "dispatched")
        self.assertEqual(dispatch.tick(self.root, self.github, self.now + timedelta(seconds=1)),
                         "awaiting-dispatch-visibility")
        self.github.dispatch.assert_called_once()

    def test_another_local_process_holds_the_lock(self):
        with (self.root / "dispatch.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(dispatch.tick(self.root, self.github, self.now), "locked")
        self.github.runs.assert_not_called()
        self.github.dispatch.assert_not_called()

    def test_api_outage_never_dispatches_without_checking_active_work(self):
        self.github.runs.side_effect = OSError("API unavailable")
        with self.assertRaises(OSError):
            dispatch.tick(self.root, self.github, self.now)
        self.github.dispatch.assert_not_called()
        self.assertFalse((self.root / "receipt.json").exists())

    def test_token_rejects_shared_permissions_and_symlinks(self):
        token = self.root / "token"
        token.write_text("test-only-token\n")
        token.chmod(0o644)
        with self.assertRaises(ValueError):
            dispatch.GitHub(token)
        token.chmod(0o600)
        client = dispatch.GitHub(token)
        link = self.root / "link"
        link.symlink_to(token)
        with self.assertRaises(ValueError):
            dispatch.GitHub(link)
        with patch.object(client, "request") as request:
            client.dispatch()
        request.assert_called_once_with(dispatch.WORKFLOW + "/dispatches",
                                        {"ref": "develop", "inputs": {"force": "false"}})


if __name__ == "__main__":
    unittest.main()
