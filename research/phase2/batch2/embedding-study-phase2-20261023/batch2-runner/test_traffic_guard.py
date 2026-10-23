"""Offline regression tests for the copied fail-closed traffic guard."""
import os
import types
import unittest
from unittest.mock import patch

from traffic_guard import Interference, TrafficGuard, discover, parse_metrics


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.guard = TrafficGuard(lambda kind, **kw: self.events.append((kind, kw)), idle_seconds=0)
        self.state = {"workers": {123: {"port": 5000, "counters": {"n_decode_total": 5}, "busy": False}}, "routers": {}}
        self.guard.snapshot = lambda: self.state

    def tearDown(self):
        self.guard.pool.shutdown()

    def test_idle_and_short_call_between_polls(self):
        self.guard.poll()
        self.assertTrue(self.guard.ready())
        self.guard.begin()
        self.state = {"workers": {123: {"port": 5000, "counters": {"n_decode_total": 6}, "busy": False}}, "routers": {}}
        self.guard.poll()
        with self.assertRaises(Interference):
            self.guard.check()

    def test_only_bound_process_terminated(self):
        calls = []
        process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("own"))
        self.guard.bind(types.SimpleNamespace(process=process))
        self.state["workers"][123]["busy"] = True
        self.guard.poll()
        self.assertEqual(calls, ["own"])
        self.assertFalse(self.guard.ready())

    def test_unavailable_fails_closed(self):
        self.guard.snapshot = lambda: (_ for _ in ()).throw(RuntimeError("unreachable"))
        self.guard.poll()
        self.assertFalse(self.guard.ready())

    def test_hung_own_shutdown_escalates_after_two_seconds(self):
        calls = []
        process = types.SimpleNamespace(poll=lambda: None, terminate=lambda: calls.append("term"),
                                        kill=lambda: calls.append("kill"))
        self.guard.bind(types.SimpleNamespace(process=process))
        self.state["workers"][123]["busy"] = True
        self.guard.poll()
        self.guard.terminate_started -= 3
        self.state = {"workers": {}, "routers": {}}
        self.guard.poll()
        self.assertIn("kill", calls)

    def test_quiet_window(self):
        self.guard.idle_seconds = 30
        self.guard.poll()
        self.assertFalse(self.guard.ready())
        self.guard.last_activity -= 31
        self.assertTrue(self.guard.ready())

    def test_end_scrape_detects_short_call(self):
        self.guard.poll()
        self.guard.begin()
        self.state = {"workers": {}, "routers": {}}
        with self.assertRaises(Interference):
            self.guard.finish()

    def test_discovery_excludes_own_child(self):
        output = f"11 {os.getpid()} /usr/bin/llama-server --port 1111\n12 1 /usr/bin/llama-server --port 2222\n13 1 /usr/bin/llama-server --models-preset x --port 8080\n14 1 /bin/sh -c llama-server --port 9999\n"
        with patch("traffic_guard.subprocess.run", return_value=types.SimpleNamespace(stdout=output)):
            workers, routers = discover()
        self.assertEqual(workers, {12: 2222})
        self.assertEqual(routers, {13: 8080})

    def test_metrics_requires_safety_fields(self):
        with self.assertRaises(ValueError):
            parse_metrics("llamacpp:n_decode_total 5\n")


if __name__ == "__main__":
    unittest.main()
