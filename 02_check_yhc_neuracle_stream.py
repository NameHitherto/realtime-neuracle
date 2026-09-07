from __future__ import annotations

import argparse
import socket
import time


EXPECTED_FIELDS = 65
BYTES_PER_FIELD = 4


def receive_for(sock: socket.socket, seconds: float) -> int:
    deadline = time.perf_counter() + seconds
    total = 0
    while time.perf_counter() < deadline:
        payload = sock.recv(1_048_576)
        if payload == b"":
            raise ConnectionError("Neuracle closed the TCP connection")
        total += len(payload)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure the Neuracle TCP byte width before starting YHC inference."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8712)
    parser.add_argument("--sampling-rate", type=float, default=1000.0)
    parser.add_argument("--measure-sec", type=float, default=5.0)
    args = parser.parse_args()
    if args.sampling_rate <= 0 or args.measure_sec <= 0:
        parser.error("sampling rate and measure duration must be > 0")

    print(f"Connecting to Neuracle TCP {args.host}:{args.port} ...")
    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        sock.settimeout(3.0)
        receive_for(sock, 1.0)
        started = time.perf_counter()
        total_bytes = receive_for(sock, args.measure_sec)
        elapsed = time.perf_counter() - started

    bytes_per_second = total_bytes / elapsed
    estimated_fields = bytes_per_second / (args.sampling_rate * BYTES_PER_FIELD)
    expected_bytes_per_second = (
        args.sampling_rate * EXPECTED_FIELDS * BYTES_PER_FIELD
    )
    print(f"Measured: {bytes_per_second:.1f} bytes/s")
    print(f"Expected: {expected_bytes_per_second:.1f} bytes/s")
    print(f"Estimated 4-byte fields/sample: {estimated_fields:.2f}")

    tolerance = EXPECTED_FIELDS * 0.15
    if abs(estimated_fields - EXPECTED_FIELDS) > tolerance:
        print(
            "FAILED: the sender is not streaming 64 EEG channels + 1 TRG field.\n"
            "Open Neuracle Data Sending and select the complete 64-channel montage plus TRG."
        )
        return 1

    print(
        "PASSED: TCP byte rate is compatible with 64 EEG + TRG at the configured sampling rate.\n"
        "Channel names/order must still match configs/neusen_w_64_channels_template.txt."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
