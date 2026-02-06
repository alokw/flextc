#!/usr/bin/env python3
"""
RLTC Encoder CLI - Generate countdown timecode audio files.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rltc import RLTCEncoder, parse_duration


def main():
    parser = argparse.ArgumentParser(
        description="Generate an RLTC countdown audio file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 5m -o countdown.wav
  %(prog)s 1:30 -r 25 -o 90sec.wav
  %(prog)s 30s --rate 50 --output test.wav

Duration formats:
  100        = 100 milliseconds
  30s        = 30 seconds
  5m         = 5 minutes
  1h         = 1 hour
  1:30       = 1 minute 30 seconds
  1:05:30    = 1 hour 5 minutes 30 seconds
        """,
    )
    parser.add_argument(
        "duration",
        type=str,
        help="Duration (e.g., '5m', '30s', '1:30', '5000ms')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="rltc_countdown.wav",
        help="Output WAV file path (default: rltc_countdown.wav)",
    )
    parser.add_argument(
        "-r", "--rate",
        type=float,
        default=30.0,
        help="Packet rate in Hz (default: 30)",
    )
    parser.add_argument(
        "-a", "--amplitude",
        type=float,
        default=0.7,
        help="Amplitude 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (default: 44100)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Parse duration
    try:
        duration_ms = parse_duration(args.duration)
    except Exception as e:
        print(f"Error parsing duration: {e}", file=sys.stderr)
        sys.exit(1)

    if args.verbose:
        duration_sec = duration_ms / 1000
        print(f"Generating {duration_sec:.1f}s countdown audio...")
        print(f"  Output: {args.output}")
        print(f"  Sample rate: {args.sample_rate} Hz")
        print(f"  Packet rate: {args.rate} Hz")
        print(f"  Amplitude: {args.amplitude}")

    # Create encoder and generate file
    encoder = RLTCEncoder(sample_rate=args.sample_rate)

    try:
        encoder.generate_to_file(
            output_path=args.output,
            duration_ms=duration_ms,
            packet_rate_hz=args.rate,
            amplitude=args.amplitude,
        )
        print(f"Generated {args.output}")
    except Exception as e:
        print(f"Error generating file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
