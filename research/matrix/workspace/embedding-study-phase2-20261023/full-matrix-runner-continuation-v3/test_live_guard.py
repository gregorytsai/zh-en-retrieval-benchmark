import time
import unittest
from unittest.mock import patch

from traffic_guard import Interference, TrafficGuard


class GuardEvidenceTests(unittest.TestCase):
    def test_changed_counter_invalidates_unit_and_emits_evidence(self):
        events=[]
        guard=TrafficGuard(lambda kind,**fields:events.append((kind,fields)),idle_seconds=30)
        idle={'workers':{1:{'busy':False,'counters':{'n_decode_total':2}}},'routers':{}}
        busy={'workers':{1:{'busy':False,'counters':{'n_decode_total':3}}},'routers':{}}
        try:
            with patch.object(guard,'snapshot',return_value=idle):
                guard.poll()
                guard.last_activity=time.monotonic()-31
                guard.begin()
            with patch.object(guard,'snapshot',return_value=busy):
                guard.poll()
            with self.assertRaises(Interference):guard.check()
            self.assertEqual(events[0][0],'guard_unit_started')
            self.assertEqual(events[0][1]['snapshot'],idle)
            self.assertTrue(any(e[0]=='external_activity' for e in events))
        finally:guard.close()

    def test_unavailable_monitor_cannot_start_or_accept(self):
        guard=TrafficGuard(lambda *a,**k:None)
        try:
            with patch.object(guard,'snapshot',side_effect=TimeoutError('metrics timeout')):
                guard.poll()
            self.assertFalse(guard.ready())
            with self.assertRaises(Interference):guard.begin()
            with self.assertRaises(Interference):guard.check()
        finally:guard.close()


if __name__=='__main__':unittest.main()
