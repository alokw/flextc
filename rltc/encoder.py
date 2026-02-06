"""
RLTC Encoder - Generates FSK-modulated audio with countdown timecode.
"""

import array
import math
import wave
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf

from . import SAMPLE_RATE, MARK_FREQ, SPACE_FREQ, BAUD_RATE, PACKET_RATE
from .packet import Packet


class RLTCEncoder:
    """
    FSK encoder for RLTC audio generation.

    Uses BFSK (Binary Frequency Shift Keying) with continuous phase.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        mark_freq: int = MARK_FREQ,
        space_freq: int = SPACE_FREQ,
        baud_rate: int = BAUD_RATE,
    ):
        """
        Initialize encoder.

        Args:
            sample_rate: Output audio sample rate (Hz)
            mark_freq: Frequency for logic 1 (Hz)
            space_freq: Frequency for logic 0 (Hz)
            baud_rate: Bit rate (bits per second)
        """
        self.sample_rate = sample_rate
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.baud_rate = baud_rate

        # Samples per bit
        self.samples_per_bit = sample_rate / baud_rate

        # Phase accumulator for continuous phase
        self.phase = 0.0

    def reset_phase(self):
        """Reset phase accumulator."""
        self.phase = 0.0

    def _generate_bit(self, bit: int, duration_bits: float = 1.0) -> np.ndarray:
        """
        Generate FSK modulated audio for one or more bits.

        Uses continuous phase FSK to avoid clicking and reduce bandwidth.

        Args:
            bit: 0 or 1
            duration_bits: Duration in bits (default 1.0)

        Returns:
            Array of audio samples (-1.0 to 1.0)
        """
        freq = self.mark_freq if bit else self.space_freq
        num_samples = int(self.samples_per_bit * duration_bits)

        # Angular frequency
        omega = 2 * math.pi * freq / self.sample_rate

        # Generate samples with continuous phase
        samples = np.zeros(num_samples)
        for i in range(num_samples):
            samples[i] = math.sin(self.phase)
            self.phase += omega
            # Wrap phase to prevent overflow (not strictly necessary but good practice)
            self.phase %= (2 * math.pi)

        return samples

    def _encode_packet(self, packet: Packet) -> np.ndarray:
        """
        Encode a single packet to FSK audio samples.

        Args:
            packet: Packet to encode

        Returns:
            Array of audio samples
        """
        data = packet.encode_with_preamble()
        samples = []

        # Convert bytes to bits (MSB first)
        for byte in data:
            for i in range(7, -1, -1):
                bit = (byte >> i) & 1
                samples.extend(self._generate_bit(bit))

        return np.array(samples)

    def generate(
        self,
        duration_ms: int,
        packet_rate_hz: float = PACKET_RATE,
        amplitude: float = 0.7,
    ) -> tuple[np.ndarray, int]:
        """
        Generate RLTC audio for a countdown duration.

        Args:
            duration_ms: Total countdown duration in milliseconds
            packet_rate_hz: Packets per second (default 30)
            amplitude: Output amplitude (0.0 to 1.0)

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        self.reset_phase()

        # Calculate timing
        interval_ms = 1000.0 / packet_rate_hz

        # Generate packets
        all_samples = []
        packet_counter = 0

        for time_ms in range(0, duration_ms, int(interval_ms)):
            # Create packet
            remaining = duration_ms - time_ms
            packet = Packet(remaining, packet_counter & 0xFFFF)

            # Encode packet
            packet_samples = self._encode_packet(packet)
            all_samples.append(packet_samples)

            packet_counter += 1

        # Concatenate all packets (they will be back-to-back)
        result = np.concatenate(all_samples)

        # Apply amplitude
        result = result * amplitude

        return result, self.sample_rate

    def generate_to_file(
        self,
        output_path: Union[str, Path],
        duration_ms: int,
        packet_rate_hz: float = PACKET_RATE,
        amplitude: float = 0.7,
    ):
        """
        Generate and save RLTC audio to file.

        Args:
            output_path: Output WAV file path
            duration_ms: Total countdown duration in milliseconds
            packet_rate_hz: Packets per second
            amplitude: Output amplitude (0.0 to 1.0)
        """
        samples, sample_rate = self.generate(duration_ms, packet_rate_hz, amplitude)

        # Save using soundfile (supports various formats)
        sf.write(
            str(output_path),
            samples,
            sample_rate,
            subtype='PCM_16'
        )


def parse_duration(duration_str: str) -> int:
    """
    Parse duration string to milliseconds.

    Formats:
    - "100" -> 100 ms
    - "5s" -> 5000 ms
    - "2m" -> 120000 ms
    - "1h" -> 3600000 ms
    - "1:30" -> 90000 ms (MM:SS)
    - "1:05:30" -> 3810000 ms (HH:MM:SS)

    Args:
        duration_str: Duration string

    Returns:
        Duration in milliseconds
    """
    duration_str = duration_str.strip().lower()

    # Try HH:MM:SS or MM:SS format
    if ":" in duration_str:
        parts = duration_str.split(":")
        if len(parts) == 2:
            # MM:SS
            minutes = int(parts[0])
            seconds = int(parts[1])
            return (minutes * 60 + seconds) * 1000
        elif len(parts) == 3:
            # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return (hours * 3600 + minutes * 60 + seconds) * 1000

    # Try unit suffixes
    if duration_str.endswith("ms"):
        return int(duration_str[:-2])
    elif duration_str.endswith("s"):
        return int(duration_str[:-1]) * 1000
    elif duration_str.endswith("m"):
        return int(duration_str[:-1]) * 60000
    elif duration_str.endswith("h"):
        return int(duration_str[:-1]) * 3600000
    else:
        # Assume seconds if no unit
        return int(duration_str) * 1000
