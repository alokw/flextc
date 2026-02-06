"""
RLTC Decoder - Real-time decodes FSK timecode from audio input.
"""

import array
import queue
import threading
import time
from collections import deque
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
from scipy import signal

from . import SAMPLE_RATE, MARK_FREQ, SPACE_FREQ, BAUD_RATE
from .packet import Packet


class FSKDemodulator:
    """
    FSK demodulator using energy-based frequency detection with bit synchronization.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        mark_freq: int = MARK_FREQ,
        space_freq: int = SPACE_FREQ,
        baud_rate: int = BAUD_RATE,
    ):
        """
        Initialize demodulator.

        Args:
            sample_rate: Audio sample rate (Hz)
            mark_freq: Frequency for logic 1 (Hz)
            space_freq: Frequency for logic 0 (Hz)
            baud_rate: Bit rate (bits per second)
        """
        self.sample_rate = sample_rate
        self.mark_freq = mark_freq
        self.space_freq = space_freq
        self.baud_rate = baud_rate

        self.samples_per_bit = int(sample_rate / baud_rate)

        # Generate Goertzel filters for each frequency
        self.mark_coeff = 2 * np.cos(2 * np.pi * mark_freq / sample_rate)
        self.space_coeff = 2 * np.cos(2 * np.pi * space_freq / sample_rate)

        # Buffer for incoming samples
        self.buffer = np.zeros(0)

        # Bit synchronization state
        self.bit_offset = 0  # Current offset within a bit period
        self.synced = False

    def _goertzel_energy(self, samples: np.ndarray, coeff: float) -> float:
        """
        Compute energy at a specific frequency using Goertzel algorithm.
        """
        n = len(samples)
        if n < 2:
            return 0.0

        s_prev = 0.0
        s_prev2 = 0.0

        for sample in samples:
            s = sample + coeff * s_prev - s_prev2
            s_prev2 = s_prev
            s_prev = s

        energy = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
        return energy

    def _find_bit_offset(self, samples: np.ndarray) -> int:
        """
        Find the optimal bit offset by looking for alternating pattern.

        Try different offsets and return the one that gives the cleanest
        alternating pattern (expected for preamble).
        """
        if len(samples) < self.samples_per_bit * 48:
            return 0

        best_offset = 0
        best_score = -1

        # Try a few offsets around the current one
        for offset in range(max(0, self.bit_offset - 5), min(self.samples_per_bit, self.bit_offset + 6)):
            bits = []
            pos = offset
            for i in range(48):  # Check first 48 bits (preamble + sync)
                if pos + self.samples_per_bit >= len(samples):
                    break
                bit_samples = samples[pos:pos + self.samples_per_bit]
                mark_energy = self._goertzel_energy(bit_samples, self.mark_coeff)
                space_energy = self._goertzel_energy(bit_samples, self.space_coeff)

                if mark_energy > space_energy:
                    bits.append(1)
                else:
                    bits.append(0)
                pos += self.samples_per_bit

            # Score by how well it matches alternating pattern
            score = 0
            for i in range(min(32, len(bits))):
                if bits[i] == (1 - i % 2):  # Should be 1, 0, 1, 0, ...
                    score += 1

            if score > best_score:
                best_score = score
                best_offset = offset

        return best_offset

    def reset(self):
        """Reset demodulator state."""
        self.buffer = np.zeros(0)
        self.bit_offset = 0
        self.synced = False

    def process(self, samples: np.ndarray) -> list[int]:
        """
        Process incoming audio samples and extract bits.

        Args:
            samples: Input audio samples (float32, -1.0 to 1.0)

        Returns:
            List of decoded bits (0 or 1)
        """
        # Add to buffer
        self.buffer = np.concatenate([self.buffer, samples])

        bits = []

        # Auto-sync if not synchronized
        if not self.synced and len(self.buffer) >= self.samples_per_bit * 50:
            self.bit_offset = self._find_bit_offset(self.buffer)
            self.synced = True

        # Process bits starting from current offset
        start_pos = self.bit_offset

        while start_pos + self.samples_per_bit <= len(self.buffer):
            bit_samples = self.buffer[start_pos:start_pos + self.samples_per_bit]

            # Compute energy at each frequency
            mark_energy = self._goertzel_energy(bit_samples, self.mark_coeff)
            space_energy = self._goertzel_energy(bit_samples, self.space_coeff)

            # Normalize by signal power
            signal_power = np.mean(bit_samples ** 2)
            if signal_power > 0.001:  # Valid signal
                if mark_energy > space_energy:
                    bits.append(1)
                else:
                    bits.append(0)

            start_pos += self.samples_per_bit

        # Remove processed samples (keep offset for next time)
        if start_pos > self.bit_offset:
            self.buffer = self.buffer[start_pos:]

        return bits


class PacketDetector:
    """
    Detects and validates RLTC packets from a bit stream.
    """

    # State machine states
    STATE_SEARCHING = 0
    STATE_FOUND_PREAMBLE = 1
    STATE_COLLECTING_PACKET = 2

    def __init__(self):
        """Initialize packet detector."""
        self.state = self.STATE_SEARCHING
        self.bit_buffer: deque[int] = deque(maxlen=256)
        self.packet_bits: list[int] = []
        self.bits_needed = 0

        # Statistics
        self.preambles_found = 0
        self.packets_found = 0

    def reset(self):
        """Reset detector state."""
        self.state = self.STATE_SEARCHING
        self.bit_buffer.clear()
        self.packet_bits = []
        self.bits_needed = 0

    def _check_alternating(self, bits: list[int], length: int = 32) -> bool:
        """
        Check if bits form an alternating pattern (101010...).
        """
        if len(bits) < length:
            return False

        errors = 0
        max_errors = length // 8  # Allow ~12.5% error rate

        for i in range(length):
            expected = 1 - (i % 2)  # 1, 0, 1, 0, ...
            actual = bits[i]
            if actual != expected:
                errors += 1
                if errors > max_errors:
                    return False

        return True

    def _check_sync_word(self, bits: list[int]) -> bool:
        """
        Check if bits match the sync word 0x16A3.
        """
        sync_bits = [0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1]

        if len(bits) < len(sync_bits):
            return False

        errors = 0
        max_errors = 5  # Allow up to 5 bit errors (31%)

        for i, expected in enumerate(sync_bits):
            if bits[i] != expected:
                errors += 1
                if errors > max_errors:
                    return False

        return True

    def _bytes_from_bits(self, bits: list[int]) -> bytes:
        """Convert list of bits to bytes (MSB first)."""
        if len(bits) % 8 != 0:
            raise ValueError("Bits must be multiple of 8")

        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)

        return bytes(result)

    def _decode_packet(self) -> Optional[Packet]:
        """Decode collected packet bits and return Packet if valid."""
        if len(self.packet_bits) < 64:
            return None

        try:
            # packet_bits contains: time (32) + counter (16) + crc (16) = 64 bits = 8 bytes
            # We need to prepend the sync word (2 bytes) to make 10 bytes
            packet_bytes = bytes([0x16, 0xA3]) + self._bytes_from_bits(self.packet_bits[:64])
            packet = Packet.decode(packet_bytes)
            return packet
        except Exception:
            return None

    def feed_bits(self, bits: list[int]) -> list[Packet]:
        """
        Feed bits to detector and return any complete packets.
        """
        packets = []

        for bit in bits:
            if self.state == self.STATE_COLLECTING_PACKET:
                # Don't add to buffer - we're collecting payload bits
                self.packet_bits.append(bit)
                self.bits_needed -= 1

                if self.bits_needed <= 0:
                    packet = self._decode_packet()
                    if packet:
                        self.packets_found += 1
                        packets.append(packet)

                    self.state = self.STATE_SEARCHING
                    self.packet_bits = []
            else:
                # In searching state, add to buffer for preamble detection
                self.bit_buffer.append(bit)

                if self.state == self.STATE_SEARCHING:
                    if len(self.bit_buffer) >= 48:
                        bits_list = list(self.bit_buffer)
                        if self._check_alternating(bits_list[:32]):
                            if self._check_sync_word(bits_list[32:48]):
                                self.preambles_found += 1
                                self.state = self.STATE_COLLECTING_PACKET
                                self.bits_needed = 64
                                self.packet_bits = []
                                for _ in range(48):
                                    if self.bit_buffer:
                                        self.bit_buffer.popleft()

        return packets


class RLTCDecoder:
    """
    Real-time RLTC decoder from audio input.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        callback: Optional[Callable[[int], None]] = None,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.callback = callback

        self.demodulator = FSKDemodulator(sample_rate)
        self.detector = PacketDetector()

        self.last_packet_counter: Optional[int] = None
        self.last_time_ms: Optional[int] = None
        self.time_history: deque[int] = deque(maxlen=5)

        self.stream: Optional[sd.InputStream] = None
        self.device = device

        self.packets_received = 0
        self.packets_invalid = 0
        self.last_update_time: Optional[float] = None

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        if status:
            print(f"Audio status: {status}")

        samples = indata.astype(np.float32).flatten()
        bits = self.demodulator.process(samples)

        if bits:
            packets = self.detector.feed_bits(bits)
            for packet in packets:
                self._handle_packet(packet)

    def _handle_packet(self, packet: Packet):
        self.packets_received += 1

        if self.last_packet_counter is not None:
            expected = (self.last_packet_counter + 1) & 0xFFFF
            if packet.packet_counter != expected:
                pass

        self.last_packet_counter = packet.packet_counter
        self.time_history.append(packet.time_remaining_ms)

        if len(self.time_history) >= 3:
            time_ms = int(np.median(list(self.time_history)))
        else:
            time_ms = packet.time_remaining_ms

        if self.last_time_ms is not None:
            if time_ms > self.last_time_ms + 5000:
                return

        self.last_time_ms = time_ms
        self.last_update_time = time.time()

        if self.callback:
            self.callback(time_ms)

    def start(self):
        if self.stream is not None:
            return

        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=0,
        )

        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_time_remaining(self) -> Optional[int]:
        if self.last_update_time is None:
            return None

        if time.time() - self.last_update_time > 0.5:
            return None

        return self.last_time_ms

    def get_statistics(self) -> dict:
        return {
            "packets_received": self.packets_received,
            "packets_invalid": self.packets_invalid,
            "time_remaining": self.get_time_remaining(),
        }


def decode_file(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
) -> list[tuple[float, int]]:
    """
    Decode RLTC from an audio file (for testing).
    """
    import soundfile as sf

    samples, sr = sf.read(file_path)

    if sr != sample_rate:
        num_samples = int(len(samples) * sample_rate / sr)
        samples = signal.resample(samples, num_samples)

    demodulator = FSKDemodulator(sample_rate)
    detector = PacketDetector()

    # Process entire file at once for proper synchronization
    bits = demodulator.process(samples)
    packets = detector.feed_bits(bits)

    # Since we process all at once, we can't easily track time position
    # Return packets with estimated time based on position
    results = []
    samples_per_packet = demodulator.samples_per_bit * 112  # 112 bits per packet
    for i, packet in enumerate(packets):
        time_sec = (i * samples_per_packet) / sample_rate
        results.append((time_sec, packet.time_remaining_ms))

    return results
