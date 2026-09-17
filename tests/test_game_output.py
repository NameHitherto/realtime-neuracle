import json
import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path

from game_output import ControlPublisher, TCPJSONOutlet


class RecordingOutlet:
    def __init__(self):
        self.samples = []
        self.fail = False

    def push(self, value):
        if self.fail:
            raise OSError('peer disconnected')
        self.samples.append((time.perf_counter(), value))

    def have_consumers(self):
        return True


class OutputTests(unittest.TestCase):
    def test_watchdog_cadence_expiry_guard_recovery_and_shutdown(self):
        outlet = RecordingOutlet()
        fresh = [True]
        pub = ControlPublisher(outlet, 30, lambda: fresh[0])
        pub.start()
        try:
            pub.submit(0, 0.15)
            time.sleep(0.25)  # acquisition/inference deliberately do nothing
            self.assertIn(0, [s[1] for s in outlet.samples])
            self.assertEqual(outlet.samples[-1][1], 3)
            self.assertGreaterEqual(len(outlet.samples), 5)
            pub.submit(2, 1)
            fresh[0] = False
            time.sleep(0.08)
            self.assertEqual(outlet.samples[-1][1], 3)
            fresh[0] = True
            time.sleep(0.08)
            self.assertEqual(outlet.samples[-1][1], 3)  # no stale resurrection
            pub.submit(1, 1)
            time.sleep(0.08)
            self.assertEqual(outlet.samples[-1][1], 1)
        finally:
            pub.stop()
        self.assertEqual([v for _, v in outlet.samples[-3:]], [3, 3, 3])

    def test_failed_output_never_replays_old_decision(self):
        outlet = RecordingOutlet()
        pub = ControlPublisher(outlet)
        pub.submit(2, 10)
        outlet.fail = True
        pub._send()
        self.assertIsNotNone(pub.last_error)
        outlet.fail = False
        pub._send()
        self.assertEqual(pub.last_sent, 3)

    def test_real_tcp_bytes_and_reconnect_requires_new_decision(self):
        with socket.socket() as server, tempfile.TemporaryDirectory() as tmp:
            server.bind(('127.0.0.1', 0))
            server.listen()
            server.settimeout(2)
            path = Path(tmp) / 'profile.json'
            profile = dict(confirmed_by_organizer=True, role='client', host='127.0.0.1', port=server.getsockname()[1], framing='newline',
                           messages={v: {'command':v} for v in ['left','right','forward','stop']})
            path.write_text(json.dumps(profile))
            outlet = TCPJSONOutlet(path)
            old = time.perf_counter()
            self.assertEqual(outlet.push(2, old), 3)
            conn, _ = server.accept()
            with conn:
                conn.settimeout(2)
                self.assertEqual(conn.recv(1024), b'{"command":"stop"}\n')
                for control, command in enumerate(['left','right','forward','stop']):
                    self.assertEqual(outlet.push(control, time.perf_counter()), control)
                    self.assertEqual(conn.recv(1024), ('{"command":"'+command+'"}\n').encode())
            outlet.close()
            outlet._next_connect = 0
            self.assertEqual(outlet.push(1, old), 3)
            conn, _ = server.accept()
            with conn:
                self.assertEqual(conn.recv(1024), b'{"command":"stop"}\n')
            outlet.close()
            profile['framing'] = 'length-prefix-be'
            path.write_text(json.dumps(profile))
            payload = TCPJSONOutlet(path).encode(0)
            self.assertEqual(struct.unpack('!I', payload[:4])[0], len(payload)-4)
            self.assertEqual(json.loads(payload[4:]), {'command':'left'})
            profile['confirmed_by_organizer'] = False
            path.write_text(json.dumps(profile))
            with self.assertRaises(ValueError):
                TCPJSONOutlet(path)

    def test_lsl_real_inlet_receives_four_float32_codes(self):
        from pylsl import StreamInlet, resolve_byprop
        from realtime_common import LSLControlOutlet
        import uuid
        name = 'BCI_ACCEPTANCE_' + uuid.uuid4().hex
        outlet = LSLControlOutlet(stream_name=name, source_id=name)
        found = resolve_byprop('name', name, timeout=3)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].channel_count(), 1)
        inlet = StreamInlet(found[0])
        inlet.open_stream(timeout=3)
        try:
            for value in range(4):
                outlet.push(value)
                sample, _ = inlet.pull_sample(timeout=2)
                self.assertEqual(sample, [float(value)])
            self.assertTrue(outlet.have_consumers())
        finally:
            inlet.close_stream()


class RuntimeContractTests(unittest.TestCase):
    def test_competition_rejects_manual_api_and_replay(self):
        from dashboard_service import InferenceRuntime, create_parser, validate_runtime_args
        args = create_parser().parse_args([])
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.args = args
        with self.assertRaises(PermissionError):
            runtime.manual_control(2)
        args.eeg_source = 'local'
        with self.assertRaises(ValueError):
            validate_runtime_args(args)
        args = create_parser().parse_args(['--default-control','2'])
        with self.assertRaises(ValueError):
            validate_runtime_args(args)

    def test_gap_discards_partial_frame_and_history_before_reconnect(self):
        from dashboard_service import InferenceRuntime, create_parser
        from telemetry_core import TelemetryBus
        from collections import deque
        class SilentServer:
            def poll(self):
                raise TimeoutError()
        runtime = InferenceRuntime.__new__(InferenceRuntime)
        runtime.args = create_parser().parse_args(['--data-timeout-sec','0.02'])
        runtime.bus = TelemetryBus()
        runtime._stop = threading.Event()
        runtime._smoother = deque([1,2,3])
        runtime.publisher = ControlPublisher(RecordingOutlet())
        runtime.logger = None
        runtime.eeg_stream_channels = ['C3']
        runtime.model_input_available = True
        runtime._calibrated = True
        with self.assertRaises(ConnectionError):
            runtime._run_connected_source(SilentServer(), 4000)
        self.assertEqual(runtime.current_control, 3)
        self.assertEqual(list(runtime._smoother), [])
        self.assertTrue(runtime._stream_faulted)


if __name__ == '__main__':
    unittest.main()
