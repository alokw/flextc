"""
SMPTE/LTC Decoder with Countdown Support

Decodes SMPTE/LTC audio and detects whether it's counting up or down.
Supports both standard SMPTE timecode and FlexTC countdown mode.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from flextc.smpte_packet import Timecode, FrameRate
from flextc.biphase import BiphaseMDecoder

# Module-level logger
_logger = logging.getLogger(__name__)


class OSCBroadcaster:
    """
    OSC broadcaster for sending timecode data.

    Broadcasts timecode as a string in the format "HH:MM:SS:FF" or "HH:MM:SS;FF" (for drop-frame).
    Uses the path /flextc/ltc or /flextc/count based on direction.
    """

    def __init__(self, address: str = "255.255.255.255", port: int = 9988):
        """
        Initialize OSC broadcaster.

        Args:
            address: IP address or hostname for OSC broadcast (default: 255.255.255.255)
            port: UDP port for OSC (default: 9988)
        """
        self.address = address
        self.port = port
        self._client = None
        self._enabled = False

    def enable(self):
        """Enable OSC broadcasting."""
        try:
            from pythonosc import udp_client
            self._client = udp_client.UDPClient(self.address, self.port)
            self._enabled = True
            _logger.info(f"OSC broadcasting enabled to {self.address}:{self.port}")
        except ImportError:
            _logger.warning("python-osc not installed. Install with: pip install python-osc")
            self._enabled = False
        except Exception as e:
            _logger.error(f"Failed to initialize OSC client: {e}")
            self._enabled = False

    def disable(self):
        """Disable OSC broadcasting and close connection."""
        self._enabled = False
        if self._client is not None:
            try:
                # python-osc UDPClient doesn't have an explicit close method
                # Just clear the reference
                self._client = None
            except Exception:
                pass

    def send_timecode(self, tc: Timecode):
        """
        Send timecode via OSC.

        Args:
            tc: Timecode object to broadcast
        """
        if not self._enabled or self._client is None:
            return

        try:
            # Format timecode as string (HH:MM:SS:FF or HH:MM:SS;FF for drop-frame)
            separator = ";" if tc.is_drop_frame else ":"
            tc_string = f"{tc.hours:02d}:{tc.minutes:02d}:{tc.seconds:02d}{separator}{tc.frames:02d}"

            # Determine OSC path based on direction
            path = "/flextc/ltc" if tc.count_up else "/flextc/count"

            # Send OSC message
            from pythonosc import udp_client
            # Build and send OSC message
            builder = udp_client.OscMessageBuilder(address=path)
            builder.add_arg(tc_string)
            msg = builder.build()
            self._client.send(msg)

        except Exception as e:
            # Don't spam logs - only log once per error type ideally
            # For now, just log on error
            _logger.debug(f"OSC send failed: {e}")


class Decoder:
    """
    SMPTE/LTC decoder with countdown detection.

    Decodes audio frames and extracts timecode information.
    Automatically detects count-up vs countdown mode.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_rate: float = 30.0,
        callback: Optional[callable] = None,
        device: Optional[int] = None,
        channel: int = 0,
        debug: bool = False,
    ):
        """
        Initialize decoder.

        Args:
            sample_rate: Audio sample rate (Hz)
            frame_rate: Frame rate (fps) - None for auto-detect
            callback: Optional callback for each frame (timecode: Timecode)
            device: Audio input device (None = default)
            channel: Audio channel to listen to (0 = first/left, 1 = second/right)
            debug: Enable debug logging
        """
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.callback = callback
        self.channel = channel
        self.debug = debug

        if self.debug:
            logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            _logger.setLevel(logging.DEBUG)
            _logger.debug(f"Decoder initialized: sample_rate={sample_rate}, frame_rate={frame_rate}, channel={channel}")

        # Biphase decoder - created when frame rate is known
        self.biphase: Optional[BiphaseMDecoder] = None

        # Buffer for auto-detection (used when frame_rate is None or signal is lost)
        self._detection_buffer: List[np.ndarray] = []
        self._detection_buffer_samples = 0
        self._detection_threshold = sample_rate // 10  # 0.1 seconds of audio for detection
        self._signal_loss_threshold = sample_rate * 0.5  # 500ms without data = signal lost (less aggressive)
        self._last_valid_sample_time: Optional[float] = None
        self._in_detection_mode = (frame_rate is None)
        self._last_signal_was_strong = False  # Track if we had strong signal in previous callback

        if frame_rate is not None:
            self.biphase = BiphaseMDecoder(sample_rate, frame_rate)
            self._last_valid_sample_time = time.time()

        # State tracking
        self.current_timecode: Optional[Timecode] = None
        self.packets_received = 0
        self._last_timecode_value = None  # For detecting stuck frames
        self._stuck_frame_count = 0  # Consecutive repeats of same timecode
        self._last_valid_hour: Optional[int] = None  # For hour continuity checking
        self._hour_confidence: int = 0  # Number of consecutive frames with same hour range

        # Audio stream
        self.stream: Optional[sd.InputStream] = None
        self.device = device

        # Last update time
        self.last_update_time: Optional[float] = None

    def _is_hour_plausible(self, hours: int) -> bool:
        """
        Check if an hour value is plausible given the history.

        At high hour values (100+), bit errors in the upper bits can cause
        large jumps. This check rejects obviously invalid hour values.

        IMPORTANT: This should only reject CLEARLY impossible values.
        Legitimate timecode jumps (like seeking) should always be allowed.

        Args:
            hours: The hour value to check

        Returns:
            True if the hour is plausible, False if it's likely a bit error
        """
        if self._last_valid_hour is None:
            # No history, accept any valid value
            return True

        # Calculate the difference
        hour_diff = abs(hours - self._last_valid_hour)

        # With 4-bit tens encoding (max 159), max single-bit error is +/- 60 hours
        # A jump of more than 80 hours is definitely a bit error
        if hour_diff > 80:
            if self.debug:
                _logger.debug(f"Hour jump > 80 (likely bit error): {self._last_valid_hour} -> {hours}")
            return False

        # For high hours (80+), check for specific bit error patterns
        # With 4-bit tens: bits 54-55 store upper 2 bits, bits 56-57 store lower 2 bits
        # Single bit flips in upper 2 bits cause jumps of 40, 60, or combinations
        if self._last_valid_hour >= 80:
            # Check for specific bit error patterns in 4-bit tens encoding
            # - Bit 54 flip (bit 2 of tens): +/- 40 hours
            # - Bit 55 flip (bit 3 of tens): +/- 60 hours
            # - Bit 56 flip (bit 0 of tens): +/- 10 hours
            # - Bit 57 flip (bit 1 of tens): +/- 20 hours
            bit_error_patterns = (10, 20, 30, 40, 50, 60, 70)
            if hour_diff in bit_error_patterns:
                # Check if units digit is the same (suggests only tens bits affected)
                if hours % 10 == self._last_valid_hour % 10:
                    if self.debug:
                        _logger.debug(f"Hour jump with same units (likely bit error): {self._last_valid_hour} -> {hours} (diff: {hour_diff})")
                    return False

        return True

    def _process_frames(self, frames: List[List[int]]) -> tuple[bool, int, int]:
        """
        Process decoded frames and extract timecode.

        Args:
            frames: List of 80-bit frame representations

        Returns:
            Tuple of (timecode_progressed: bool, valid_timecode_count: int, frames_rejected_for_hour: int)
        """
        timecode_progressed = False
        valid_timecode_count = 0
        frames_rejected_for_hour = 0  # Track frames rejected due to hour check

        for frame_bits in frames:
            try:
                timecode = Timecode.decode_80bit(frame_bits)
            except Exception as e:
                # Skip invalid frames - could be bit errors in the stream
                if self.debug:
                    _logger.debug(f"Failed to decode frame: {e}")
                continue

            if timecode:
                # Check if the hour is plausible given our history
                if not self._is_hour_plausible(timecode.hours):
                    # Skip this frame - likely a bit error
                    frames_rejected_for_hour += 1
                    if self.debug:
                        _logger.debug(f"Skipping frame with implausible hour: {timecode.hours}")
                    continue

                valid_timecode_count += 1
                # Check if timecode is stuck (repeating same value)
                tc_value = (timecode.hours, timecode.minutes, timecode.seconds, timecode.frames)
                if self._last_timecode_value == tc_value:
                    self._stuck_frame_count += 1
                else:
                    self._stuck_frame_count = 0
                    self._last_timecode_value = tc_value
                    timecode_progressed = True

                self.current_timecode = timecode
                self.packets_received += 1
                self.last_update_time = time.time()

                # Update hour confidence tracking
                if self._last_valid_hour is not None:
                    if self._last_valid_hour // 10 == timecode.hours // 10:
                        # Same tens range, increase confidence
                        self._hour_confidence += 1
                    else:
                        # Hour tens changed, reset confidence
                        self._hour_confidence = 1
                else:
                    self._hour_confidence = 1
                self._last_valid_hour = timecode.hours

                if self.callback:
                    self.callback(timecode)

        return timecode_progressed, valid_timecode_count, frames_rejected_for_hour

    def _audio_callback(self, indata: np.ndarray, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        try:
            self._audio_callback_impl(indata, frames, time_info, status)
        except Exception as e:
            # Log error but don't crash - audio callback must not raise exceptions
            _logger.error(f"Error in audio callback: {e}")
            # Reset state to recover
            self._in_detection_mode = True
            self.biphase = None
            self.current_timecode = None

    def _audio_callback_impl(self, indata: np.ndarray, frames, time_info, status):
        """Implementation of audio callback - wrapped in try/except by _audio_callback."""
        if status:
            _logger.warning(f"Audio status: {status}")

        # Extract the specified channel
        if indata.shape[1] > self.channel:
            samples = indata[:, self.channel].astype(np.float32)
        else:
            # Channel not available, use first channel
            samples = indata[:, 0].astype(np.float32)

        current_time = time.time()
        current_signal_max = np.max(np.abs(samples))

        if self.debug:
            _logger.debug(f"Callback: frames={len(samples)}, signal_max={current_signal_max:.4f}, "
                          f"in_detection={self._in_detection_mode}, biphase={'None' if self.biphase is None else 'exists'}")

        # SIMPLIFIED STATE MACHINE:
        # - Detection mode: accumulating samples to find frame rate
        # - Locked mode: have frame rate, processing timecodes
        # On ANY anomaly (signal loss, discontinuity), go back to detection mode

        # Check for signal loss - 500ms timeout (less aggressive for stability)
        if (not self._in_detection_mode and self.biphase is not None and
            self._last_valid_sample_time is not None and
            current_time - self._last_valid_sample_time > 0.5):
            # Signal lost - immediately enter detection mode
            self._in_detection_mode = True
            self.biphase = None
            self.frame_rate = None
            self.current_timecode = None
            self._last_valid_sample_time = None
            self._stuck_frame_count = 0
            self._last_timecode_value = None
            self._last_valid_hour = None  # Reset hour tracking to allow new hour values
            self._hour_confidence = 0
            self._detection_buffer.clear()
            self._detection_buffer_samples = 0
            # Don't log warning every time - too noisy
            if self.debug:
                _logger.warning("[SIGNAL LOSS] Entering detection mode")
            # Fall through to detection mode

        # If in detection mode, accumulate samples for frame rate detection
        if self._in_detection_mode or self.biphase is None:

            # Track signal state transitions
            current_is_strong = current_signal_max >= 0.01
            signal_returned = current_is_strong and not self._last_signal_was_strong

            if signal_returned:
                # Signal just returned - purge any stale silent data from buffer
                self._detection_buffer.clear()
                self._detection_buffer_samples = 0
                _logger.info(f"[SIGNAL RETURN] Signal detected after silence, clearing stale buffer")

            self._last_signal_was_strong = current_is_strong

            # Check signal level of current block before adding to buffer
            if not current_is_strong:
                # Current block is essentially silent, skip it and reset
                self._detection_buffer.clear()
                self._detection_buffer_samples = 0
                return

            self._detection_buffer.append(samples)
            self._detection_buffer_samples += len(samples)

            # Limit buffer size to prevent runaway accumulation
            max_buffer_size = self._detection_threshold * 20  # Max 2 seconds
            if self._detection_buffer_samples > max_buffer_size:
                # Keep only the most recent samples
                self._detection_buffer = [self._detection_buffer[-1]]
                self._detection_buffer_samples = len(self._detection_buffer[-1])
                _logger.warning(f"Detection buffer overflow, truncating to {self._detection_buffer_samples} samples")

            # Check if we have enough samples for detection
            # Use a lower threshold for initial detection attempt, then increase
            min_samples = self._detection_threshold // 2  # Start with 0.05 seconds

            if self._detection_buffer_samples >= min_samples:
                # Combine buffer and detect frame rate
                combined = np.concatenate(self._detection_buffer)

                # Check signal level of combined buffer
                signal_max = np.max(np.abs(combined))
                if signal_max < 0.01:
                    # Signal too weak, clear buffer and keep accumulating fresh samples
                    self._detection_buffer.clear()
                    self._detection_buffer_samples = 0
                    if self.debug:
                        _logger.debug(f"Detection: combined signal too weak ({signal_max:.4f}), clearing buffer")
                    return

                # Try with progressively larger buffers until we find valid timecodes
                # or hit the maximum buffer size
                if self._detection_buffer_samples >= self._detection_threshold * 2:
                    # We've accumulated 2x the threshold without success - try harder
                    # Skip the first portion to try different alignments
                    skip_amount = len(combined) // 4
                    for offset in [0, skip_amount, skip_amount * 2, skip_amount * 3]:
                        test_buffer = combined[offset:] if offset > 0 else combined

                        detected_rate = detect_frame_rate(test_buffer, self.sample_rate)
                        self.frame_rate = detected_rate
                        self.biphase = BiphaseMDecoder(self.sample_rate, detected_rate)

                        # Process buffered samples
                        decoded_frames = self.biphase.process(test_buffer)

                        # Check if we got any frames with valid sync (even if timecode is invalid)
                        # This is more lenient than requiring valid timecodes
                        valid_frame_found = False
                        for frame_bits in decoded_frames:
                            if len(frame_bits) >= 80:
                                # Check for valid sync pattern (bits 64-79 must start with 0011)
                                sync_bits = frame_bits[64:68]
                                if sync_bits == [0, 0, 1, 1]:
                                    valid_frame_found = True
                                    if self.debug:
                                        _logger.debug(f"Detection: found valid sync at offset {offset}")
                                    break

                        if valid_frame_found:
                            # Try to get at least one valid timecode
                            self._process_frames(decoded_frames)
                            # If we got any valid timecode, exit detection mode
                            if self.current_timecode is not None:
                                self._in_detection_mode = False
                                self._last_valid_sample_time = current_time
                                self._stuck_frame_count = 0
                                self._last_timecode_value = None
                                _logger.info(f"[DETECTION] Successfully locked on signal! (offset: {offset})")
                                # Clear buffer after success
                                self._detection_buffer.clear()
                                self._detection_buffer_samples = 0
                                return

                    # None of the offsets worked - clear some buffer and try again
                    self._detection_buffer.clear()
                    self._detection_buffer_samples = 0
                    self.biphase = None
                    self.frame_rate = None
                    _logger.warning("[DETECTION] No valid frames found at any offset, retrying...")
                    return

                # Normal detection path - try with current buffer
                detected_rate = detect_frame_rate(combined, self.sample_rate)
                self.frame_rate = detected_rate
                self.biphase = BiphaseMDecoder(self.sample_rate, detected_rate)
                _logger.info(f"[DETECTION] Detected frame rate: {detected_rate} fps")

                # Process buffered samples
                decoded_frames = self.biphase.process(combined)

                # Try to get at least one valid timecode (more lenient - just need valid sync)
                valid_frame_found = False
                for frame_bits in decoded_frames:
                    if len(frame_bits) >= 80:
                        sync_bits = frame_bits[64:68]
                        if sync_bits == [0, 0, 1, 1]:
                            valid_frame_found = True
                            if self.debug:
                                _logger.debug(f"Detection: found valid sync")
                            break

                if valid_frame_found:
                    # Try to process frames - may get valid timecodes
                    self._process_frames(decoded_frames)
                    # If we got any valid timecode, exit detection mode
                    if self.current_timecode is not None:
                        self._in_detection_mode = False
                        self._last_valid_sample_time = current_time
                        self._stuck_frame_count = 0
                        self._last_timecode_value = None
                        _logger.info("[DETECTION] Successfully locked on signal!")
                        # Clear buffer after success
                        self._detection_buffer.clear()
                        self._detection_buffer_samples = 0
                else:
                    # No valid timecodes yet - keep accumulating, don't clear buffer
                    # Stay in detection mode but keep the decoder
                    if self.debug:
                        _logger.debug(f"Detection: no valid timecodes yet, buffer size: {self._detection_buffer_samples}")
                return
            else:
                return  # Still accumulating

        # Decode to frames - but skip if signal is too weak
        # This prevents the decoder buffer from getting corrupted with silence
        current_signal_max = np.max(np.abs(samples))

        if current_signal_max < 0.01:
            # Signal too weak - don't process these samples
            return

        decoded_frames = self.biphase.process(samples)

        # Process frames
        if decoded_frames:
            timecode_progressed, valid_count, rejected_for_hour = self._process_frames(decoded_frames)

            # If we're getting frames but rejecting some for hour issues, still consider
            # the signal valid - update _last_valid_sample_time to avoid signal loss
            if valid_count > 0 or rejected_for_hour > 0:
                self._last_valid_sample_time = current_time

            # Check for stuck frames - same timecode repeating without progress
            # If we see the same frame 5+ times with strong signal, decoder is corrupted
            if (not self._in_detection_mode and
                self._stuck_frame_count >= 5 and
                current_signal_max >= 0.05):
                # Reset biphase decoder to clear corrupted state
                self.biphase.reset()
                self._stuck_frame_count = 0
                self._last_timecode_value = None
                _logger.warning("[STUCK FRAMES] Resetting biphase decoder to recover")
                return

            # Check for frames but no valid timecodes - decoder is misaligned
            if valid_count == 0 and not self._in_detection_mode and current_signal_max >= 0.05:
                # Track consecutive invalid frame decodes
                if not hasattr(self, '_invalid_frame_count'):
                    self._invalid_frame_count = 0
                self._invalid_frame_count += 1

                # After MORE consecutive callbacks with frames but no valid timecodes, reset
                # Increased from 3 to 10 to be more tolerant at high hours where bit errors
                # in the hour field are more common
                if self._invalid_frame_count >= 10:
                    self.biphase.reset()
                    self._stuck_frame_count = 0
                    self._last_timecode_value = None
                    self._invalid_frame_count = 0
                    _logger.warning("[INVALID FRAMES] Frames produced but no valid timecodes, resetting biphase decoder")
                return

            # Reset invalid frame counter on valid decode
            if valid_count > 0:
                if hasattr(self, '_invalid_frame_count'):
                    self._invalid_frame_count = 0
        else:
            # No frames decoded even though signal is strong
            # This could mean the biphase decoder is corrupted or misaligned
            # Instead of entering detection mode (which loses frame rate context),
            # just reset the biphase decoder and let it re-sync
            if not self._in_detection_mode and current_signal_max >= 0.05:
                # Track consecutive empty decodes
                if not hasattr(self, '_empty_decode_count'):
                    self._empty_decode_count = 0
                self._empty_decode_count += 1

                # After 3 consecutive empty decodes with strong signal, reset the decoder
                if self._empty_decode_count >= 3:
                    self.biphase.reset()
                    self._stuck_frame_count = 0
                    self._last_timecode_value = None
                    self._empty_decode_count = 0
                    _logger.warning("[NO FRAMES] Resetting biphase decoder to recover")
                return

            # Reset empty decode counter on successful decode
            if hasattr(self, '_empty_decode_count'):
                self._empty_decode_count = 0
        # Note: We DON'T update _last_valid_sample_time on empty decodes
        # This means 500ms with no valid frames = signal loss = enter detection mode

    def start(self):
        """Start decoding from audio input."""
        if self.stream is not None:
            return  # Already running

        # Request all available channels so user can select which one to use
        self.stream = sd.InputStream(
            device=self.device,
            channels=2,  # Request stereo to get both channels
            samplerate=self.sample_rate,
            callback=self._audio_callback,
            blocksize=0,
        )

        self.stream.start()

    def stop(self):
        """Stop decoding."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_timecode(self) -> Optional[Timecode]:
        """
        Get the current timecode.

        Returns:
            Timecode object, or None if no valid data received recently
        """
        current_time = time.time()

        # Consider data stale if no updates for 1 second
        if self.last_update_time is None:
            return None

        if current_time - self.last_update_time > 1.0:
            return None

        return self.current_timecode

    def get_statistics(self) -> dict:
        """
        Get decoder statistics.

        Returns:
            Dict with current status
        """
        return {
            "packets_received": self.packets_received,
            "timecode": self.get_timecode(),
        }


def format_timecode(tc: Timecode) -> str:
    """Format timecode for display (with ; separator for drop-frame)."""
    if tc is None:
        return "--:--:--:--"

    direction = "▲" if tc.count_up else "▼"
    separator = ";" if tc.is_drop_frame else ":"
    return f"{direction} {tc.hours:02d}:{tc.minutes:02d}:{tc.seconds:02d}{separator}{tc.frames:02d}"


def detect_frame_rate(
    samples: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Auto-detect frame rate from audio samples by trying all rates.

    The detection works by:
    1. Trying each candidate frame rate with the biphase decoder
    2. Counting how many valid timecodes are produced
    3. Checking which frame rate's biphase timing produces the most consistent results
    4. Also tries inverted polarity for compatibility with different encoders

    Args:
        samples: Audio samples
        sample_rate: Sample rate in Hz

    Returns:
        Detected frame rate, or 30.0 as default
    """
    # Frame rates to try with their identifying characteristics
    # (fps_value, bit10_drop, bit11_color)
    frame_rate_configs = [
        (30.0, False, False),
        (25.0, False, True),
        (29.97, True, False),   # 29.97 drop-frame
        (24.0, False, False),
        (23.98, False, False),
        (29.97, False, False),  # 29.97 non-drop
        (30.0, True, False),    # 30 drop-frame
    ]

    best_rate = 30.0
    best_score = -1

    # Try both normal and inverted polarity
    for inverted in (False, True):
        test_samples = -samples if inverted else samples

        for test_rate, bit10_drop, bit11_color in frame_rate_configs:
            decoder = BiphaseMDecoder(sample_rate, test_rate)
            frames = decoder.process(test_samples.copy())

            if not frames:
                continue

            # Analyze the decoded frames
            matching_bits = 0
            valid_timecodes = 0

            for i in range(min(len(frames), 50)):
                frame_bits = frames[i]
                if len(frame_bits) < 80:
                    continue

                # Check if bits 10 and 11 match what we expect for this rate
                actual_bit10 = frame_bits[10]
                actual_bit11 = frame_bits[11]

                if actual_bit10 == (1 if bit10_drop else 0) and actual_bit11 == (1 if bit11_color else 0):
                    matching_bits += 1

                # Also count valid timecodes
                tc = Timecode.decode_80bit(frame_bits)
                if tc:
                    valid_timecodes += 1

            # Score: prioritize matching bits, then valid timecode count
            # Inverted gets slightly lower score to prefer normal polarity when equal
            polarity_penalty = 0 if not inverted else 10
            score = matching_bits * 100 + valid_timecodes - polarity_penalty

            if score > best_score:
                best_score = score
                best_rate = test_rate

    return best_rate


def decode_file(
    file_path: str,
    sample_rate: int = 44100,
    frame_rate: float = None,
) -> tuple:
    """
    Decode SMPTE/LTC from an audio file.

    Reads the beginning and end of the file to determine start/end timecodes
    and actual direction (by comparing timecodes).

    Args:
        file_path: Path to audio file
        sample_rate: Expected sample rate
        frame_rate: Frame rate (None for auto-detect)

    Returns:
        Tuple of (first_timecode, last_timecode, detected_frame_rate, file_duration_seconds)
    """
    # Read the entire file to get total length (soundfile is fast for this)
    all_samples, sr = sf.read(file_path)
    total_samples = len(all_samples)
    total_duration = total_samples / sr

    # Use first channel if stereo
    if len(all_samples.shape) > 1:
        all_samples = all_samples[:, 0]

    if sr != sample_rate:
        # Resample if needed
        from scipy import signal
        all_samples = signal.resample(all_samples, int(len(all_samples) * sample_rate / sr))
        sr = sample_rate

    # Auto-detect frame rate and polarity from the beginning if not specified
    start_samples = all_samples[:int(sr * 2)]
    if frame_rate is None:
        frame_rate = detect_frame_rate(start_samples, sr)

    # Detect polarity by trying both and seeing which produces MORE valid timecodes
    use_inverted = False
    test_decoder = BiphaseMDecoder(sr, frame_rate)
    test_frames_normal = test_decoder.process(start_samples.copy())
    valid_count_normal = sum(1 for fb in test_frames_normal if Timecode.decode_80bit(fb))

    test_decoder.reset()
    test_frames_inverted = test_decoder.process(-start_samples.copy())
    valid_count_inverted = sum(1 for fb in test_frames_inverted if Timecode.decode_80bit(fb))

    # Use the polarity that produces more valid timecodes
    if valid_count_inverted > valid_count_normal:
        use_inverted = True

    # Apply detected polarity if needed
    if use_inverted:
        all_samples = -all_samples

    # Process the file in chunks to maintain biphase decoder sync
    decoder = BiphaseMDecoder(sr, frame_rate)
    chunk_size = int(sr * 1)  # 1 second chunks

    all_timecodes = []
    for i in range(0, len(all_samples), chunk_size):
        chunk = all_samples[i:i+chunk_size]
        frames = decoder.process(chunk)
        for fb in frames:
            tc = Timecode.decode_80bit(fb)
            if tc:
                all_timecodes.append(tc)

    first_tc = all_timecodes[0] if all_timecodes else None
    last_tc = all_timecodes[-1] if all_timecodes else None

    return first_tc, last_tc, frame_rate, total_duration


def main():
    parser = argparse.ArgumentParser(
        description="Decode SMPTE/LTC timecode from audio input.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i test.wav                  # Decode from file
  %(prog)s                              # Decode from live audio
  %(prog)s -d 2                         # Use device 2
  %(prog)s --list-devices               # Show audio devices

The decoder automatically detects:
- Frame rate (from timing analysis, or use -r to specify)
- Count-up mode (standard SMPTE, shows ▲)
- Countdown mode (countdown, shows ▼)

Direction is indicated by bit 60:
- Bit 60 = 1: Counting up (standard SMPTE)
- Bit 60 = 0: Counting down (countdown mode)
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
        "-c", "--channel",
        type=int,
        default=0,
        help="Audio channel to listen to (0=left, 1=right, default: 0)",
    )
    parser.add_argument(
        "-s", "--sample-rate",
        type=int,
        default=48000,
        help="Sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "-r", "--frame-rate",
        type=float,
        default=None,
        choices=[23.98, 24.0, 25.0, 29.97, 30.0],
        help="Frame rate (default: auto-detect)",
    )
    parser.add_argument(
        "-l", "--list-devices",
        action="store_true",
        help="List available audio input devices",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output with detailed logging",
    )
    parser.add_argument(
        "--osc",
        action="store_true",
        help="Enable OSC broadcasting of timecode",
    )
    parser.add_argument(
        "--osc-address",
        type=str,
        default="255.255.255.255",
        help="OSC broadcast address (default: 255.255.255.255)",
    )
    parser.add_argument(
        "--osc-port",
        type=int,
        default=9988,
        help="OSC UDP port (default: 9988)",
    )

    args = parser.parse_args()

    if args.list_devices:
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

        first_tc, last_tc, detected_rate, duration = decode_file(
            args.input, args.sample_rate, args.frame_rate
        )

        if first_tc is None:
            print("No SMPTE/LTC frames detected.", file=sys.stderr)
            sys.exit(1)

        # Show detected frame rate
        if args.frame_rate is None:
            print(f"Auto-detected frame rate: {detected_rate} fps")
            print()

        # Determine actual direction by comparing first and last timecodes
        # (ignoring bit 60, since that may not be set correctly in standard LTC)
        fps = first_tc.fps

        def tc_to_frames(tc):
            total = (tc.hours * 3600 + tc.minutes * 60 + tc.seconds) * fps + tc.frames
            return int(total)

        first_frames = tc_to_frames(first_tc)
        if last_tc is not None:
            last_frames = tc_to_frames(last_tc)
            # Determine direction from actual timecode values
            counting_up = last_frames >= first_frames
        else:
            # No end timecode found, fall back to bit 60
            counting_up = first_tc.count_up
            last_frames = first_frames

        direction_text = "Counting up" if counting_up else "Counting down"
        direction_symbol = "▲" if counting_up else "▼"

        # Calculate duration from timecodes
        if counting_up:
            duration_frames = last_frames - first_frames
        else:
            duration_frames = first_frames - last_frames

        # Convert duration frames to HH:MM:SS:FF
        dur_hours = int(duration_frames // (fps * 3600))
        duration_frames %= fps * 3600
        dur_minutes = int(duration_frames // (fps * 60))
        duration_frames %= fps * 60
        dur_seconds = int(duration_frames // fps)
        dur_ff = int(duration_frames % fps)

        # Use ; separator for drop-frame timecodes
        separator = ";" if first_tc.is_drop_frame else ":"
        tc_duration = f"{dur_hours:02d}:{dur_minutes:02d}:{dur_seconds:02d}{separator}{dur_ff:02d}"

        # Display summary
        print(f"Start timecode: {format_timecode(first_tc)}")
        if last_tc is not None:
            print(f"End timecode:   {direction_symbol} {last_tc.hours:02d}:{last_tc.minutes:02d}:{last_tc.seconds:02d}{separator}{last_tc.frames:02d}")
        else:
            print("End timecode:   (not found in file)")
        print(f"Duration:       {tc_duration}")
        print(f"Direction:      {direction_text}")
        print(f"Frame rate:     {detected_rate} fps")
        return

    # Live decoding mode
    print("Decoding SMPTE/LTC from live audio input...")
    if args.device is not None:
        print(f"Using device {args.device}")
    print(f"Using channel {args.channel} ({'left' if args.channel == 0 else 'right'})")
    if args.frame_rate is None:
        print("Frame rate: auto-detecting...")
    else:
        print(f"Frame rate: {args.frame_rate} fps")
    print("Press Ctrl+C to stop.")
    print("-" * 40)

    # Initialize OSC broadcaster if enabled
    osc_broadcaster = None
    if args.osc:
        osc_broadcaster = OSCBroadcaster(address=args.osc_address, port=args.osc_port)
        osc_broadcaster.enable()
        print(f"OSC broadcasting: {args.osc_address}:{args.osc_port}")
        print(f"  Paths: /flextc/ltc, /flextc/count")
        print("-" * 40)

    decoder = Decoder(
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        device=args.device,
        channel=args.channel,
        debug=args.verbose,
    )

    def display_callback(tc: Timecode):
        pass  # We'll display in the main loop

    # Track detection state for display
    last_was_detecting = False
    detection_shown = args.frame_rate is not None

    try:
        decoder.start()

        last_display = ""
        while True:
            time.sleep(0.1)

            tc = decoder.get_timecode()
            stats = decoder.get_statistics()

            # Send via OSC if enabled
            if tc and osc_broadcaster:
                osc_broadcaster.send_timecode(tc)

            # Check if we're in detection mode
            is_detecting = decoder._in_detection_mode or decoder.biphase is None

            # Show detection complete message when we first lock on
            if not is_detecting and last_was_detecting:
                print(f"\nAuto-detected frame rate: {decoder.frame_rate} fps")
                print("-" * 40)
                detection_shown = True

            if tc:
                display = f"\r{format_timecode(tc)}  (frames: {stats['packets_received']})  "
                print(display, end="", flush=True)
                last_display = display
            else:
                if is_detecting:
                    # Still detecting or re-detecting after signal loss
                    waiting_msg = f"\rDetecting frame rate... ({int(decoder._detection_buffer_samples / decoder.sample_rate * 10) / 10}s / 0.1s)  "
                    print(waiting_msg, end="", flush=True)
                    last_was_detecting = True
                elif time.time() - (decoder.last_update_time or 0) > 0.5:
                    print(f"\r{' ' * 60}", end="", flush=True)
                    print(f"\r--:--:--:--  (waiting for signal...)  ", end="", flush=True)

            last_was_detecting = is_detecting

    except KeyboardInterrupt:
        print("\n\nStopped.")
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        decoder.stop()
        if osc_broadcaster:
            osc_broadcaster.disable()


if __name__ == "__main__":
    main()
