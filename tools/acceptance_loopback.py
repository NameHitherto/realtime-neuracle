"""Synthetic TCP EEG -> actual checkpoint -> LSL. Not a human/BCI accuracy test."""
import json
import socket
import struct
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from pylsl import StreamInlet, resolve_byprop
from dashboard_service import InferenceRuntime, create_parser
from telemetry_core import TelemetryBus


def main():
    root = Path('outputs/final-debug/loopback')
    root.mkdir(parents=True, exist_ok=True)
    halt = threading.Event()
    pause = threading.Event()
    connection_times = []
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        listener.settimeout(0.2)
        def feed():
            while not halt.is_set():
                try:
                    conn, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                with conn:
                    conn.settimeout(0.3)
                    connection_times.append(time.monotonic())
                    counter = 0
                    while not halt.is_set():
                        if pause.is_set():
                            halt.wait(0.05)
                            continue
                        t = (np.arange(100)+counter)/1000
                        channels = np.arange(64)[:, None]
                        eeg = (20*np.sin(2*np.pi*(10+channels/20)*t+channels)*np.cos(channels)).astype('float32')
                        packet = b''.join(struct.pack('<64fi', *eeg[:, i], 0) for i in range(100))
                        try:
                            conn.sendall(packet)
                        except OSError:
                            break
                        counter += 100
                        halt.wait(0.1)
        thread = threading.Thread(target=feed, daemon=True)
        thread.start()
        name = 'BCI_SYNTHETIC_' + uuid.uuid4().hex
        args = create_parser().parse_args(['--cpu','--port',str(listener.getsockname()[1]),
            '--stream-name',name,'--rate-check-sec','0.5','--live-baseline-sec','0.5',
            '--min-confidence','0','--reconnect-sec','0.2','--log-dir',str(root), '--experiment-id','SYNTHETIC_NOT_HUMAN'])
        bus = TelemetryBus()
        runtime = InferenceRuntime(args, bus)
        runtime.start()
        inlet = None
        samples = []
        try:
            found = resolve_byprop('name', name, timeout=5)
            assert found, 'Output stream not found'
            inlet = StreamInlet(found[0])
            inlet.open_stream(timeout=3)
            def collect(seconds):
                deadline = time.monotonic()+seconds
                while time.monotonic() < deadline:
                    sample, ts = inlet.pull_sample(timeout=0.2)
                    if sample is not None:
                        samples.append((time.monotonic(), sample[0], ts))
            collect(8)
            first_predictions = runtime.logger.total_predictions
            assert first_predictions > 0, 'Real checkpoint did not infer'
            pause.set()
            paused_at = time.monotonic()
            collect(4)
            late_pause = [s[1] for s in samples if s[0] > paused_at+2.2]
            assert late_pause and all(x == 3 for x in late_pause), 'Output did not stop on EEG gap'
            before_recovery = runtime.logger.total_predictions
            resumed_at = time.monotonic()
            pause.clear()
            collect(2.5)
            assert runtime.logger.total_predictions == before_recovery, 'Inference used pre-gap EEG'
            assert all(s[1] == 3 for s in samples if s[0] > resumed_at), 'Old decision replayed after reconnect'
            collect(5)
            assert runtime.logger.total_predictions > before_recovery, 'Inference failed to recover'
            runtime.stop()
            collect(0.3)
            report = {'kind':'synthetic loopback, real checkpoint; no human EEG or official game execution',
                      'passed':True,'samples':len(samples),'observed_codes': sorted(set(s[1] for s in samples)),
                      'predictions_before_gap':first_predictions,'predictions_total':runtime.logger.total_predictions,
                      'connections':len(connection_times),'gap_stop_passed':True,'fresh_window_recovery_passed':True,
                      'session_dir':str(runtime.logger.session_dir)}
            (root/'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            (root/'received-samples.json').write_text(json.dumps(samples), encoding='utf-8')
            print(json.dumps(report, indent=2))
        finally:
            runtime.stop()
            if inlet:
                inlet.close_stream()
            halt.set()
            thread.join(2)


if __name__ == '__main__':
    main()
