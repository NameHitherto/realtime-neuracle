"""Read-only LAN diagnostics: LSL discovery/listening or TCP connect-only probe."""
import argparse
import json
import os
import socket
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lsl-config', type=Path)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--listen', help='Exact LSL stream name, e.g. Unity_KartData')
    parser.add_argument('--source-id', help='Select exactly one player/output when names collide')
    parser.add_argument('--tcp-host')
    parser.add_argument('--tcp-port', type=int)
    args = parser.parse_args()
    if not 0 < args.seconds <= 900:
        parser.error('--seconds must be in (0,900]')
    if args.tcp_host:
        if not args.tcp_port or not 0 < args.tcp_port < 65536:
            parser.error('Supply the organizer-confirmed TCP port')
        with socket.create_connection((args.tcp_host, args.tcp_port), timeout=3):
            print('TCP connected. No payload sent; JSON compatibility NOT verified.')
        return
    if args.lsl_config:
        os.environ['LSLAPICFG'] = str(args.lsl_config.resolve(strict=True))
    from pylsl import resolve_streams, StreamInlet
    streams = resolve_streams(wait_time=min(args.seconds, 5))
    for stream in streams:
        print(json.dumps({'name':stream.name(), 'type':stream.type(), 'source_id':stream.source_id(),
                          'host':stream.hostname(), 'channels':stream.channel_count(), 'format':stream.channel_format(),
                          'rate':stream.nominal_srate()}, ensure_ascii=False))
    if not args.listen:
        if not streams:
            print('No visible LSL streams; verify source, network, config and firewall.')
        return
    selected = [s for s in streams if s.name() == args.listen and (not args.source_id or s.source_id() == args.source_id)]
    if len(selected) != 1:
        parser.error(f'Expected exactly one matching stream; got {len(selected)}. Use --source-id to disambiguate.')
    inlet = StreamInlet(selected[0])
    try:
        inlet.open_stream(timeout=3)
        deadline = time.monotonic()+args.seconds
        while time.monotonic() < deadline:
            sample, timestamp = inlet.pull_sample(timeout=0.2)
            if sample is not None:
                print(json.dumps({'received_at':time.time(), 'lsl_timestamp':timestamp, 'sample':sample}, ensure_ascii=False), flush=True)
    finally:
        inlet.close_stream()


if __name__ == '__main__':
    main()
