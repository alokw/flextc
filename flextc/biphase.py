"""
Biphase-M (Manchester) Encoding/Decoding for SMPTE/LTC.

Biphase-M encoding rules:
1. There's always a transition at the START of each bit cell
2. Logic 0: Additional transition in middle of cell
3. Logic 1: No transition in middle

Reference: EBU Tech 3185 / SMPTE 12M

Waveform Types:
- "square": Instantaneous transitions (fast rise/fall time, ~1μs equivalent)
- "sine": Smooth sinusoidal transitions (slower rise/fall time, ~25μs equivalent)
"""

import numpy as np
from typing import List, Tuple, Optional, Literal
from scipy import signal


WaveformType = Literal["square", "sine"]


def _square_to_sine(samples: np.ndarray, cutoff_ratio: float = 0.4) -> np.ndarray:
    """
    Convert square wave to sine-like waveform using filtering.

    This preserves the zero-crossing timing while smoothing the edges
    to create a more broadcast-friendly signal.

    Args:
        samples: Square wave samples
        cutoff_ratio: Lowpass filter cutoff relative to Nyquist (default 0.4)

    Returns:
        Smoothed sine-like waveform
    """
    # Design a Butterworth lowpass filter to smooth the square wave
    # The cutoff frequency is set to preserve the fundamental frequency
    # while attenuating harmonics that create the sharp edges
    nyquist = 0.5
    cutoff = cutoff_ratio * nyquist
    b, a = signal.butter(4, cutoff, btype='low')

    # Apply filter
    smoothed = signal.filtfilt(b, a, samples)

    # Normalize to maintain original amplitude
    max_orig = np.max(np.abs(samples))
    max_smoothed = np.max(np.abs(smoothed))
    if max_smoothed > 0:
        smoothed = smoothed * (max_orig / max_smoothed)

    return smoothed.astype(np.float32)


class BiphaseMEncoder:
    """
    Biphase-M (Manchester) encoder for SMPTE/LTC.

    Supports two waveform types:
    - square: Instant transitions (traditional for hardware LTC)
    - sine: Smooth sinusoidal transitions (broadcast-friendly, reduces harmonics)
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_rate: float = 30.0,
        waveform: WaveformType = "square",
    ):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.waveform = waveform
        self.bit_rate = 80 * frame_rate
        # Use float for 29.97 fps to maintain correct timing
        # 29.97 fps at 48kHz: 48000 / (80 * 29.97) ≈ 20.02 samples per bit
        # 30 fps at 48kHz: 48000 / (80 * 30) = 20.0 samples per bit
        self.samples_per_bit = sample_rate / self.bit_rate
        self.samples_per_bit_int = int(self.samples_per_bit)
        self.use_float_bits = (abs(frame_rate - 29.97) < 0.01 or abs(frame_rate - 23.98) < 0.01)
        self.last_level = -1  # Start low
        self.sample_accumulator = 0.0  # For handling fractional samples

    def reset(self):
        """Reset encoder state."""
        self.last_level = -1

    def encode_bit(self, bit: int) -> np.ndarray:
        """Encode a single bit to audio samples using Biphase-M."""
        # Always transition at start
        first_half_level = -self.last_level

        if bit == 0:
            # Logic 0: No middle transition (stays at same level)
            second_half_level = first_half_level
        else:
            # Logic 1: Additional transition in middle
            second_half_level = -first_half_level

        self.last_level = second_half_level

        # Calculate sample counts with accumulator for precise timing
        self.sample_accumulator += self.samples_per_bit
        total_samples = int(self.sample_accumulator)
        self.sample_accumulator -= total_samples

        half_samples = total_samples // 2
        other_half = total_samples - half_samples

        # Generate square wave (regardless of waveform type - we'll convert later)
        first_half = np.full(half_samples, first_half_level, dtype=np.float32)
        second_half = np.full(other_half, second_half_level, dtype=np.float32)
        return np.concatenate([first_half, second_half])

    def encode_frame(self, bits: List[int]) -> np.ndarray:
        """Encode 80-bit frame to audio samples."""
        if len(bits) != 80:
            raise ValueError(f"Frame must be 80 bits, got {len(bits)}")

        all_samples = []
        for bit in bits:
            samples = self.encode_bit(bit)
            all_samples.append(samples)

        frame_samples = np.concatenate(all_samples)

        # Apply waveform conversion if needed
        if self.waveform == "sine":
            frame_samples = _square_to_sine(frame_samples)

        return frame_samples

    def encode_timecode(self, timecode_frames: List[List[int]]) -> np.ndarray:
        """Encode multiple frames of timecode."""
        all_samples = []
        for frame_bits in timecode_frames:
            samples = self.encode_frame(frame_bits)
            all_samples.append(samples)

        result = np.concatenate(all_samples)

        # For sine waveform, we also need to smooth across frame boundaries
        # Apply a final filter pass to the entire signal
        if self.waveform == "sine" and len(timecode_frames) > 1:
            result = _square_to_sine(result)

        return result


class BiphaseMDecoder:
    """
    Biphase-M (Manchester) decoder for SMPTE/LTC.
    """

    def __init__(self, sample_rate: int = 44100, frame_rate: float = 30.0):
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.bit_rate = 80 * frame_rate
        # Use float for 29.97 fps to maintain correct timing
        if abs(frame_rate - 29.97) < 0.01 or abs(frame_rate - 23.98) < 0.01:
            self.samples_per_bit = sample_rate / self.bit_rate
            self.use_float_bits = True
        else:
            self.samples_per_bit = int(sample_rate / self.bit_rate)
            self.use_float_bits = False

        self.buffer = np.zeros(0)
        self.bit_buffer: List[int] = []

    def reset(self):
        """Reset decoder state."""
        self.buffer = np.zeros(0)
        self.bit_buffer = []

    def _find_edges(self, samples: np.ndarray) -> List[int]:
        """Find edge positions (zero-crossings)."""
        edges = []

        # Check if there's an implicit edge at position 0
        # The encoder always starts with a transition from -1 to 1
        if len(samples) > 0 and samples[0] > 0:
            # First sample is positive, so there was a transition to positive at position 0
            edges.append(0)

        for i in range(1, len(samples)):
            if (samples[i-1] >= 0 and samples[i] < 0) or (samples[i-1] < 0 and samples[i] >= 0):
                edges.append(i)

        return edges

    def _decode_from_edges(self, edges: List[int]) -> List[int]:
        """
        Decode bits from edge positions using Biphase-M encoding rules.

        In Biphase-M:
        - Every bit cell starts with a transition (edge)
        - Logic 0 has a single transition (at start only)
        - Logic 1 has an additional transition in the middle

        Returns as many bits as can be decoded from the given edges.
        """
        if not edges:
            return []

        bits = []
        half_period = int(round(self.samples_per_bit / 2))
        full_period = int(round(self.samples_per_bit))
        tolerance = 2

        # Walk through edges, grouping them into bit cells
        edge_idx = 0

        while edge_idx < len(edges):
            start_edge_pos = edges[edge_idx]

            # Check if there's a middle edge
            has_middle = False
            next_bit_start_idx = edge_idx + 1

            if edge_idx + 1 < len(edges):
                next_edge_pos = edges[edge_idx + 1]
                distance = next_edge_pos - start_edge_pos

                if abs(distance - half_period) <= tolerance:
                    # Next edge is approximately half_period away -> it's a middle edge
                    has_middle = True
                    # The middle edge is NOT the start of the next bit
                    # We need to find the actual start of the next bit
                    if edge_idx + 2 < len(edges):
                        next_next_edge_pos = edges[edge_idx + 2]
                        distance_to_next = next_next_edge_pos - next_edge_pos
                        if abs(distance_to_next - half_period) <= tolerance:
                            # Pattern: start, middle (9), next start (9 more)
                            # This bit is 0, next bit starts at edge_idx + 2
                            next_bit_start_idx = edge_idx + 2
                        else:
                            # Unexpected pattern, assume next edge starts next bit
                            next_bit_start_idx = edge_idx + 1
                    else:
                        # No more edges after this
                        next_bit_start_idx = edge_idx + 1
                elif abs(distance - full_period) <= tolerance:
                    # Next edge is approximately full_period away -> no middle edge
                    has_middle = False
                    next_bit_start_idx = edge_idx + 1
                else:
                    # Unexpected distance
                    has_middle = False
                    next_bit_start_idx = edge_idx + 1
            else:
                # Last edge, no more data
                has_middle = False
                next_bit_start_idx = edge_idx + 1

            bits.append(1 if has_middle else 0)
            edge_idx = next_bit_start_idx

        return bits

    def process(self, samples: np.ndarray) -> List[List[int]]:
        """Process audio samples and extract complete frames."""
        self.buffer = np.concatenate([self.buffer, samples])

        edges = self._find_edges(self.buffer)
        bits = self._decode_from_edges(edges)
        self.bit_buffer.extend(bits)

        # Trim buffer
        keep_samples = int(round(self.samples_per_bit * 10))
        if len(self.buffer) > keep_samples:
            self.buffer = self.buffer[-keep_samples:]

        # Extract complete frames
        frames = []
        attempts = 0
        max_attempts = 50  # Limit iterations to prevent infinite loops

        while len(self.bit_buffer) >= 80 and attempts < max_attempts:
            attempts += 1
            frame_found = False

            for offset in range(min(20, len(self.bit_buffer) - 79)):
                frame_bits = self.bit_buffer[offset:offset + 80]

                # Check sync (bits 64-79)
                # SMPTE/LTC sync word: 0011 1111 1111 1101 (normal polarity)
                # Inverse polarity: 1100 0000 0000 0010
                sync = frame_bits[64:80]
                sync_expected = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
                sync_expected_inv = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

                errors_normal = sum(1 for a, b in zip(sync, sync_expected) if a != b)
                errors_inv = sum(1 for a, b in zip(sync, sync_expected_inv) if a != b)

                if errors_normal <= 2:
                    frames.append(frame_bits)
                    self.bit_buffer = self.bit_buffer[offset + 80:]
                    frame_found = True
                    break
                elif errors_inv <= 2:
                    # Inverse polarity - invert all bits and add
                    frame_bits = [1 - b for b in frame_bits]
                    frames.append(frame_bits)
                    self.bit_buffer = self.bit_buffer[offset + 80:]
                    frame_found = True
                    break

            if frame_found:
                # Successfully found and extracted a frame, continue looking for more
                continue
            else:
                # No valid frame found at any offset, discard bits aggressively
                # Drop more bits when we're stuck to clear corrupted data faster
                drop_amount = min(8, len(self.bit_buffer))
                self.bit_buffer = self.bit_buffer[drop_amount:]

        # If bit_buffer is getting too large, clear it entirely (corrupted state)
        if len(self.bit_buffer) > 500:
            self.bit_buffer = []

        return frames


def verify_biphase_round_trip():
    """Test that encode/decode round trip works correctly."""
    encoder = BiphaseMEncoder(sample_rate=44100, frame_rate=30.0)
    decoder = BiphaseMDecoder(sample_rate=44100, frame_rate=30.0)

    # Test the full 80-bit frame
    test_bits = [0] * 80
    test_bits[64:80] = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]

    encoder.reset()
    samples = encoder.encode_frame(test_bits)
    frames = decoder.process(samples)

    if frames:
        match = frames[0] == test_bits
        print(f"Round trip test: {'PASS' if match else 'FAIL'}")
        if not match:
            errors = [(i, a, b) for i, (a, b) in enumerate(zip(test_bits, frames[0])) if a != b]
            for i, expected, got in errors[:10]:
                print(f"  Bit {i}: expected {expected}, got {got}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        return match
    else:
        print("Round trip test: FAIL (no frames decoded)")
        return False


if __name__ == "__main__":
    verify_biphase_round_trip()
