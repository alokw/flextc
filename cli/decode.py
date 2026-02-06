#!/usr/bin/env python3
"""
RLTC Decoder CLI - Real-time decode countdown from audio input.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rltc import RLTCDecoder, decode_file


def format_time(ms: Optional[int]) -> str:
    """Format milliseconds as HH:MM:SS.mmm"""
    if ms is None:
        return "--:--:--.---"

    seconds = ms // 1000
    milliseconds = ms % 1000

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def main():
    parser = argparse.ArgumentParser(
        description="Decode RLTC countdown from audio input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Decode from default input
  %(prog)s -d 2               # Use device 2
  %(prog)s -i test.wav        # Decode from file
  %(prog)s --list-devices     # Show audio devices
        """,
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        help="Decode from file instead of live audio",
    )
    parser.add_argument(
        "-d", "--device",
        type=int,
        help="Audio input device number (default: system default)",
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "-l", "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with statistics",
    )

    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print("Audio Input Devices:")
        print("-" * 60)
        for i, dev in enumerate(sd.query_devices()):
            if dev['max_input_channels'] > 0:
                print(f"  [{i}] {dev['name']}")
        return

    # File decoding mode
    if args.input:
        print(f"Decoding from file: {args.input}")
        print("-" * 40)

        results = decode_file(args.input, args.sample_rate)

        if not results:
            print("No RLTC packets detected.", file=sys.stderr)
            sys.exit(1)

        # Show results
        print(f"Detected {len(results)} packets:")
        for timestamp, time_ms in results[:10]:  # Show first 10
            print(f"  {timestamp:6.2f}s -> {format_time(time_ms)}")

        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more")

        # Show time range
        first_time = results[0][1]
        last_time = results[-1][1]
        print(f"\nTime range: {format_time(first_time)} to {format_time(last_time)}")
        return

    # Live decoding mode
    print("Decoding RLTC from live audio input...")
    if args.device is not None:
        print(f"Using device {args.device}")
    print("Press Ctrl+C to stop.")
    print("-" * 40)

    last_display_time = 0
    packet_count = 0
    last_time_ms = None

    def time_callback(time_ms: int):
        nonlocal last_time_ms, packet_count
        last_time_ms = time_ms
        packet_count += 1

    decoder = RLTCDecoder(
        sample_rate=args.sample_rate,
        device=args.device,
        callback=time_callback,
    )

    try:
        decoder.start()

        while True:
            time.sleep(0.1)  # Update display 10x per second

            current_time = decoder.get_time_remaining()

            if current_time is not None:
                # Update display
                time_str = format_time(current_time)
                print(f"\r {time_str}  (packets: {packet_count})", end="", flush=True)
                last_display_time = time.time()
            else:
                # No signal
                if time.time() - last_display_time > 0.5:
                    print(f"\r {format_time(None)}  (waiting for signal...)  ", end="", flush=True)
                    last_time_ms = None

            if args.verbose and packet_count > 0 and packet_count % 100 == 0:
                stats = decoder.get_statistics()
                print(f"\nStats: {stats}")

    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        decoder.stop()


if __name__ == "__main__":
    main()
