"""
SMPTE/LTC Encoder with Countdown Support

Generates SMPTE-compatible audio files with either:
- Standard timecode (counting up)
- Countdown timecode (counting down) - TTRRTT countdown mode

Both modes produce valid SMPTE/LTC that can be read by standard decoders.
The direction is indicated by bit 60 of the frame.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import soundfile as sf

from ttrrtt.smpte_packet import Timecode, FrameRate, generate_countdown, generate_countup
from ttrrtt.biphase import BiphaseMEncoder


class Encoder:
    """
    SMPTE/LTC encoder with support for countdown mode.

    Generates audio files compatible with standard SMPTE/LTC decoders.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_rate: float = 30.0,
        amplitude: float = 0.7,
        drop_frame: bool = False,
    ):
        """
        Initialize encoder.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_rate: Frame rate (fps) - 24, 25, 29.97, or 30
            amplitude: Output amplitude (0.0 to 1.0)
            drop_frame: Enable drop-frame mode (for 29.97 or 30 fps)
        """
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.amplitude = amplitude

        # Map frame rate to enum
        if abs(frame_rate - 23.98) < 0.01:
            self.frame_rate_enum = FrameRate.FPS_23_98
        elif abs(frame_rate - 24.0) < 0.1:
            self.frame_rate_enum = FrameRate.FPS_24
        elif abs(frame_rate - 25.0) < 0.1:
            self.frame_rate_enum = FrameRate.FPS_25
        elif abs(frame_rate - 29.97) < 0.01:
            # Check for drop-frame flag
            if drop_frame:
                self.frame_rate_enum = FrameRate.FPS_29_97_DROP
            else:
                self.frame_rate_enum = FrameRate.FPS_29_97_NDF
        elif frame_rate == 30.0:
            # Check for drop-frame flag
            if drop_frame:
                self.frame_rate_enum = FrameRate.FPS_30_DROP
            else:
                self.frame_rate_enum = FrameRate.FPS_30
        else:
            self.frame_rate_enum = FrameRate.FPS_30

        self.biphase = BiphaseMEncoder(sample_rate, frame_rate)

    @property
    def _fps_for_calc(self) -> int:
        """Get integer fps for frame count calculations."""
        # 23.98 and 29.97 use 24 and 30 for frame numbering (fractional only affects timing)
        if self.frame_rate_enum in (FrameRate.FPS_23_98, FrameRate.FPS_24):
            return 24
        elif self.frame_rate_enum in (FrameRate.FPS_29_97_NDF, FrameRate.FPS_29_97_DROP, FrameRate.FPS_30_DROP, FrameRate.FPS_30):
            return 30
        elif self.frame_rate_enum == FrameRate.FPS_25:
            return 25
        else:
            return 30

    def generate_countdown(
        self,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        frames: int = 0,
        start_hours: int = 0,
        start_minutes: int = 0,
        start_seconds: int = 0,
        start_frames: int = 0,
    ) -> np.ndarray:
        """
        Generate countdown audio.

        Args:
            hours: Duration in hours
            minutes: Duration in minutes
            seconds: Duration in seconds
            frames: Duration in frames
            start_hours: Starting hours (for countdown from specific time)
            start_minutes: Starting minutes
            start_seconds: Starting seconds
            start_frames: Starting frames

        Returns:
            Array of audio samples
        """
        # Calculate duration in total frames
        duration_frames = (hours * 3600 + minutes * 60 + seconds) * self._fps_for_calc + frames

        # Calculate starting timecode in total frames
        # If no start time specified, start from the duration
        if start_hours == 0 and start_minutes == 0 and start_seconds == 0 and start_frames == 0:
            start_total = duration_frames
        else:
            start_total = (start_hours * 3600 + start_minutes * 60 + start_seconds) * self._fps_for_calc + start_frames

        # Generate countdown timecodes starting from start time
        timecodes = []
        for i in range(duration_frames + 1):
            remaining = start_total - i
            if remaining >= 0:
                tc = Timecode.from_total_frames(remaining, self.frame_rate_enum, count_up=False)
                timecodes.append(tc)

        # Convert to 80-bit frames
        frames_bits = [tc.encode_80bit() for tc in timecodes]

        # Encode to audio
        samples = self.biphase.encode_timecode(frames_bits)

        return samples * self.amplitude

    def generate_countup(
        self,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        frames: int = 0,
        start_hours: int = 0,
        start_minutes: int = 0,
        start_seconds: int = 0,
        start_frames: int = 0,
    ) -> np.ndarray:
        """
        Generate count-up audio (standard SMPTE).

        Args:
            hours: Duration in hours
            minutes: Duration in minutes
            seconds: Duration in seconds
            frames: Duration in frames
            start_hours: Starting hours (for count-up from specific time)
            start_minutes: Starting minutes
            start_seconds: Starting seconds
            start_frames: Starting frames

        Returns:
            Array of audio samples
        """
        # Calculate starting timecode in total frames
        start_total = (start_hours * 3600 + start_minutes * 60 + start_seconds) * self._fps_for_calc + start_frames

        # Calculate duration in total frames
        duration_frames = (hours * 3600 + minutes * 60 + seconds) * self._fps_for_calc + frames

        # Generate count-up timecodes starting from start time
        timecodes = []
        for i in range(duration_frames + 1):
            current = start_total + i
            tc = Timecode.from_total_frames(current, self.frame_rate_enum, count_up=True)
            timecodes.append(tc)

        # Convert to 80-bit frames
        frames_bits = [tc.encode_80bit() for tc in timecodes]

        # Encode to audio
        samples = self.biphase.encode_timecode(frames_bits)

        return samples * self.amplitude

    def generate_to_file(
        self,
        output_path: Union[str, Path],
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        frames: int = 0,
        countdown: bool = False,
        start_hours: int = 0,
        start_minutes: int = 0,
        start_seconds: int = 0,
        start_frames: int = 0,
    ):
        """
        Generate and save to file.

        Args:
            output_path: Output WAV file path
            hours: Duration in hours
            minutes: Duration in minutes
            seconds: Duration in seconds
            frames: Duration in frames
            countdown: True for countdown, False for count-up (default)
            start_hours: Starting hours
            start_minutes: Starting minutes
            start_seconds: Starting seconds
            start_frames: Starting frames
        """
        if countdown:
            samples = self.generate_countdown(hours, minutes, seconds, frames, start_hours, start_minutes, start_seconds, start_frames)
        else:
            samples = self.generate_countup(hours, minutes, seconds, frames, start_hours, start_minutes, start_seconds, start_frames)

        # Save to file
        sf.write(
            str(output_path),
            samples,
            self.sample_rate,
            subtype='PCM_16'
        )


def parse_timecode(time_str: str) -> tuple:
    """
    Parse timecode string to hours, minutes, seconds, frames.

    Formats:
    - "1:30:00" -> HH:MM:SS
    - "1:30:00:15" -> HH:MM:SS:FF
    - "90s" -> 90 seconds
    - "5m" -> 5 minutes
    - "1h" -> 1 hour

    Returns:
        tuple: (hours, minutes, seconds, frames)
    """
    time_str = time_str.strip()

    # HH:MM:SS:FF format
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:
            # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            frames = 0
        elif len(parts) == 4:
            # HH:MM:SS:FF
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            frames = int(parts[3])
        else:
            raise ValueError(f"Invalid timecode format: {time_str}")
        return (hours, minutes, seconds, frames)

    # Try unit suffixes
    if time_str.endswith('s'):
        seconds = int(time_str[:-1])
        return (0, 0, seconds, 0)
    elif time_str.endswith('m'):
        minutes = int(time_str[:-1])
        return (0, minutes, 0, 0)
    elif time_str.endswith('h'):
        hours = int(time_str[:-1])
        return (hours, 0, 0, 0)
    elif time_str.endswith('f'):
        frames = int(time_str[:-1])
        return (0, 0, 0, frames)
    else:
        # Assume seconds if no unit
        seconds = int(time_str)
        return (0, 0, seconds, 0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SMPTE/LTC timecode audio (count-up or countdown).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s 1:30 -o countup.wav              # 1 minute 30 second count-up (default)
  %(prog)s 1:30:00 -o countup.wav           # Same, explicit format
  %(prog)s 5m --countdown -o countdown.wav  # 5 minute countdown
  %(prog)s 0:5:30:15 -o test.wav            # 5 min 30 sec 15 frames
  %(prog)s 10s --start 1:00:00:00 -o from_1hr.wav  # Count-up from 1 hour

Timecode formats (duration):
  1:30       = 1 minute 30 seconds
  1:30:00    = 1 minute 30 seconds
  1:30:00:15 = 1 min 30 sec 15 frames
  30s        = 30 seconds
  5m         = 5 minutes
  1h         = 1 hour

Timecode formats (--start option):
  1:00:00:00 = 1 hour
  15:30:00:00 = 15 hours 30 minutes

The generated audio is valid SMPTE/LTC that can be read by standard decoders.
For countdown mode, bit 60 indicates reverse direction.
        """,
    )
    parser.add_argument(
        "timecode",
        type=str,
        help="Duration (e.g., '1:30', '5m', '1:30:00:15')",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output WAV file path (default: auto-generated based on parameters)",
    )
    parser.add_argument(
        "-r", "--frame-rate",
        type=float,
        default=30.0,
        choices=[23.98, 24.0, 25.0, 29.97, 30.0],
        help="Frame rate (default: 30.0)",
    )
    parser.add_argument(
        "--drop-frame",
        action="store_true",
        help="Enable drop-frame mode (for 29.97 or 30 fps)",
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "-a", "--amplitude",
        type=float,
        default=0.7,
        help="Amplitude 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--countdown",
        action="store_true",
        help="Generate countdown timecode (default is count-up/standard SMPTE)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Starting timecode (e.g., '1:00:00:00', '15:45:00:00')",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Parse timecode (duration)
    try:
        hours, minutes, seconds, frames = parse_timecode(args.timecode)
    except Exception as e:
        print(f"Error parsing timecode: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse start timecode
    start_hours = start_minutes = start_seconds = start_frames = 0
    if args.start:
        try:
            start_hours, start_minutes, start_seconds, start_frames = parse_timecode(args.start)
        except Exception as e:
            print(f"Error parsing start timecode: {e}", file=sys.stderr)
            sys.exit(1)

    # Generate output filename if not specified
    if args.output:
        output_path = args.output
        if not output_path.lower().endswith('.wav'):
            output_path = output_path + '.wav'
    else:
        # Auto-generate filename: ltc_{rate}fps{_drop}_{start}_{duration}{_countdown}.wav
        # Format frame rate
        if args.frame_rate == 23.98:
            rate_str = "2398"
        elif args.frame_rate == 29.97:
            rate_str = "2997"
        else:
            rate_str = str(int(args.frame_rate))

        # Add drop suffix if applicable
        drop_suffix = "_drop" if args.drop_frame else ""

        # Format duration part
        if hours > 0:
            duration_str = f"{hours}h"
            if minutes > 0:
                duration_str += f"{minutes}m"
        elif minutes > 0:
            duration_str = f"{minutes}m"
            if seconds > 0:
                duration_str += f"{seconds}s"
        elif seconds > 0:
            duration_str = f"{seconds}s"
        elif frames > 0:
            duration_str = f"{frames}f"
        else:
            duration_str = "0s"

        # Format start timecode if specified
        start_str = ""
        if args.start:
            start_str = f"_{start_hours:02d}{start_minutes:02d}{start_seconds:02d}{start_frames:02d}"

        # Direction suffix
        direction_str = "_countdown" if args.countdown else ""

        output_path = f"ltc_{rate_str}fps{drop_suffix}{start_str}{direction_str}_{duration_str}.wav"

    if args.verbose:
        direction = "countdown" if args.countdown else "count-up"
        print(f"Generating {direction} timecode:")
        print(f"  Duration: {hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}")
        if args.start:
            print(f"  Start: {start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}:{start_frames:02d}")
        print(f"  Frame rate: {args.frame_rate} fps")
        print(f"  Output: {output_path}")
        print(f"  Sample rate: {args.sample_rate} Hz")

        # Calculate total frames and duration
        total_frames = (hours * 3600 + minutes * 60 + seconds) * int(args.frame_rate) + frames
        duration_sec = total_frames / args.frame_rate
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration_sec:.1f} seconds")

    # Create encoder and generate file
    encoder = Encoder(
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        amplitude=args.amplitude,
        drop_frame=args.drop_frame,
    )

    try:
        encoder.generate_to_file(
            output_path=output_path,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            countdown=args.countdown,
            start_hours=start_hours,
            start_minutes=start_minutes,
            start_seconds=start_seconds,
            start_frames=start_frames,
        )
        print(f"Generated {output_path}")
    except Exception as e:
        print(f"Error generating file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
