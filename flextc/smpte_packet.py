"""
SMPTE/LTC Timecode Packet Structure

Compatible with standard SMPTE timecode, with added support for countdown mode.

SMPTE/LTC uses 80 bits per frame (per SMPTE 12M / Wikipedia):
- Bits 0-3: Frame units (BCD, 0-9)
- Bits 4-7: User bits field 1
- Bits 8-9: Frame tens (BCD, 0-2)
- Bit 10: Drop frame flag (1 = 29.97 drop frame, 0 = no drop frame)
- Bit 11: Color frame flag
- Bits 12-15: User bits field 2
- Bits 16-19: Seconds units (BCD, 0-9)
- Bits 20-23: User bits field 3
- Bits 24-26: Seconds tens (BCD, 0-5)
- Bit 27: Flag (differs by frame rate: polarity correction at 30fps, BGF2 at 25fps)
- Bits 28-31: User bits field 4
- Bits 32-35: Minutes units (BCD, 0-9)
- Bits 36-39: User bits field 5
- Bits 40-42: Minutes tens (BCD, 0-5)
- Bit 43: Flag (differs by frame rate: BGF0 at 25fps, polarity/BGF at other rates)
- Bits 44-47: User bits field 6
- Bits 48-51: Hours units (BCD, 0-9)
- Bits 52-55: User bits field 7
- Bits 56-57: Hours tens (BCD, 0-2)
- Bit 58: Clock flag / Binary group flag 1 (external clock sync)
- Bit 59: Flag (differs by frame rate: BGF0 at 25fps, polarity/BGF at other rates)
- Bits 60-63: User bits field 8
  - Bit 60: FlexTC direction flag (1 = counting up, 0 = counting down)
  - Bits 61-63: User bits (part of field 8)
- Bits 64-79: Sync word (0011 1111 1111 1101)

Frame rate identification via bit 10 (drop frame) and bit 11 (color frame):
- 00 = 24 fps
- 01 = 25 fps
- 10 = 29.97 fps (drop frame)
- 11 = 30 fps

Direction flag (FlexTC extension):
- Bit 60: 1 = counting up (standard SMPTE), 0 = counting down (countdown mode)

Note on polarity bit (bit 27 at non-25fps rates):
- Chosen to provide even number of 0 bits in whole frame (including sync)
- Keeps phase consistent so each frame starts with rising edge at bit 0
"""

import struct
from typing import Optional, Literal
from dataclasses import dataclass
from enum import IntEnum


class FrameRate(IntEnum):
    """SMPTE frame rates."""
    FPS_24 = 0
    FPS_25 = 1
    FPS_29_97_NDF = 2  # 29.97 non-drop
    FPS_30 = 3
    FPS_23_98 = 4  # 24 * 1000/1001
    FPS_29_97_DROP = 5  # 29.97 drop-frame
    FPS_30_DROP = 6  # 30 drop-frame (MTC standard)


@dataclass
class Timecode:
    """
    SMPTE timecode representation.

    Can represent either elapsed time (counting up) or remaining time (counting down).
    """
    hours: int = 0
    minutes: int = 0
    seconds: int = 0
    frames: int = 0
    frame_rate: FrameRate = FrameRate.FPS_30

    # Direction flag: True = counting up (standard), False = counting down (countdown mode)
    count_up: bool = True

    # Drop frame flag (for 29.97 df)
    drop_frame: bool = False

    # User bits (8 fields of 4 bits each)
    user_bits: list[int] = None

    def __post_init__(self):
        """Validate and normalize timecode values."""
        # Note: We support extended hours encoding up to 639 hours using bits 48-57.
        # Bits 48-51: units (hours % 10)
        # Bits 52-57: tens (hours // 10), where bits 52-55 are upper bits and bits 56-57 are lower bits
        # Hours 0-39 remain fully compatible with standard SMPTE decoders (bits 52-55 = 0000).
        # Hours 40-639 use FlexTC extended mode (bits 52-55 non-zero).
        if self.hours < 0:
            raise ValueError(f"Hours cannot be negative (got {self.hours})")
        if self.hours > 639:
            raise ValueError(f"Hours cannot exceed 639 (got {self.hours})")
        self.minutes = max(0, min(self.minutes, 59))
        self.seconds = max(0, min(self.seconds, 59))

        # Max frames depends on frame rate
        max_frames = self.max_frames
        self.frames = max(0, min(self.frames, max_frames))

        if self.frame_rate == FrameRate.FPS_29_97_DROP:
            self.drop_frame = True

        if self.user_bits is None:
            self.user_bits = [0] * 8

    @property
    def fps(self) -> float:
        """Get actual frame rate as float."""
        if self.frame_rate == FrameRate.FPS_24:
            return 24.0
        elif self.frame_rate == FrameRate.FPS_25:
            return 25.0
        elif self.frame_rate == FrameRate.FPS_29_97_NDF:
            return 29.97
        elif self.frame_rate == FrameRate.FPS_23_98:
            return 23.976
        elif self.frame_rate == FrameRate.FPS_30:
            return 30.0
        elif self.frame_rate == FrameRate.FPS_29_97_DROP:
            return 29.97
        elif self.frame_rate == FrameRate.FPS_30_DROP:
            return 30.0
        else:
            return 30.0

    @property
    def max_frames(self) -> int:
        """Get maximum frames value for current frame rate."""
        if self.frame_rate == FrameRate.FPS_24:
            return 23
        elif self.frame_rate == FrameRate.FPS_25:
            return 24
        elif self.frame_rate == FrameRate.FPS_29_97_NDF:
            return 29
        elif self.frame_rate == FrameRate.FPS_23_98:
            return 23
        elif self.frame_rate == FrameRate.FPS_30:
            return 29
        elif self.frame_rate == FrameRate.FPS_29_97_DROP:
            return 29
        elif self.frame_rate == FrameRate.FPS_30_DROP:
            return 29
        else:
            return 29

    @property
    def is_drop_frame(self) -> bool:
        """Check if this timecode uses drop-frame encoding."""
        return self.frame_rate in (FrameRate.FPS_29_97_DROP, FrameRate.FPS_30_DROP)

    @property
    def total_frames(self) -> int:
        """Convert timecode to total frame count."""
        total = self.frames
        total += self.seconds * int(self.fps)
        total += self.minutes * 60 * int(self.fps)
        total += self.hours * 60 * 60 * int(self.fps)
        return total

    @classmethod
    def from_total_frames(cls, total_frames: int, frame_rate: FrameRate,
                         count_up: bool = True) -> 'Timecode':
        """Create timecode from total frame count."""
        # Map frame rate enum to integer fps value for calculation
        if frame_rate == FrameRate.FPS_24:
            fps = 24
        elif frame_rate == FrameRate.FPS_25:
            fps = 25
        elif frame_rate == FrameRate.FPS_29_97_NDF:
            fps = 30  # Use 30 for frame count purposes (29.97 is just timing)
        elif frame_rate == FrameRate.FPS_30:
            fps = 30
        elif frame_rate == FrameRate.FPS_23_98:
            fps = 24  # Use 24 for frame count purposes (23.98 is just timing)
        elif frame_rate == FrameRate.FPS_29_97_DROP:
            fps = 30  # Drop-frame affects frame numbering, not total count calculation
        elif frame_rate == FrameRate.FPS_30_DROP:
            fps = 30
        else:
            fps = 30  # Default

        hours = total_frames // (3600 * fps)
        remaining = total_frames % (3600 * fps)
        minutes = remaining // (60 * fps)
        remaining = remaining % (60 * fps)
        seconds = remaining // fps
        frames = remaining % fps

        return cls(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            frame_rate=frame_rate,
            count_up=count_up
        )

    def encode_80bit(self) -> list[int]:
        """
        Encode timecode to 80-bit SMPTE/LTC format.

        Bit positions per SMPTE standard (Wikipedia):
        - Bits 0-3: Frame units (BCD)
        - Bits 4-7: User bits field 1
        - Bits 8-9: Frame tens (BCD)
        - Bit 10: Drop frame flag
        - Bit 11: Color frame flag
        - Bits 12-15: User bits field 2
        - Bits 16-19: Seconds units (BCD)
        - Bits 20-23: User bits field 3
        - Bits 24-26: Seconds tens (BCD)
        - Bit 27: Flag (polarity correction at 30fps, BGF2 at 25fps)
        - Bits 28-31: User bits field 4
        - Bits 32-35: Minutes units (BCD)
        - Bits 36-39: User bits field 5
        - Bits 40-42: Minutes tens (BCD)
        - Bit 43: Flag (BGF0 at 25fps, polarity/BGF at other rates)
        - Bits 44-47: User bits field 6
        - Bits 48-51: Hours units (BCD)
        - Bits 52-55: User bits field 7
        - Bits 56-57: Hours tens (BCD)
        - Bit 58: Clock flag / BGF1
        - Bit 59: Flag (BGF0 at 25fps, polarity/BGF at other rates) - FlexTC direction
        - Bits 60-63: User bits field 8
        - Bits 64-79: Sync word (0011 1111 1111 1101)

        Returns:
            List of 80 bits (0 or 1)
        """
        bits = [0] * 80

        # Frame units (bits 0-3): Frames % 10 in BCD (LSB first)
        frame_units = self.frames % 10
        for i in range(4):
            bits[i] = (frame_units >> i) & 1

        # User bits field 1 (bits 4-7)
        for i in range(4):
            bits[4 + i] = (self.user_bits[0] >> i) & 1

        # Frame tens (bits 8-9): Frames // 10 in BCD
        frame_tens = self.frames // 10
        bits[8] = frame_tens & 1
        bits[9] = (frame_tens >> 1) & 1

        # Bits 10-11: Frame rate identification
        # Based on reference file analysis:
        # - Non-drop (24, 30, 29.97 NDF, 23.98): bit 10=0, bit 11=0
        # - Drop frame (29.97 DF, 30 DF): bit 10=1, bit 11=0
        # - 25 fps: bit 10=0, bit 11=1 (from Wikipedia)
        # Note: 24 fps and 30 fps both encode to 00, distinguished by timing
        # Note: 23.98 fps encodes to 00 like other non-drop rates
        if self.frame_rate in (FrameRate.FPS_29_97_DROP, FrameRate.FPS_30_DROP):
            bits[10] = 1  # Drop frame flag
        else:
            bits[10] = 0  # Non-drop

        if self.frame_rate == FrameRate.FPS_25:
            bits[11] = 1  # 25 fps indicator
        else:
            bits[11] = 0  # Not 25 fps

        # User bits field 2 (bits 12-15)
        for i in range(4):
            bits[12 + i] = (self.user_bits[1] >> i) & 1

        # Seconds units (bits 16-19): Seconds % 10 in BCD (LSB first)
        sec_units = self.seconds % 10
        for i in range(4):
            bits[16 + i] = (sec_units >> i) & 1

        # User bits field 3 (bits 20-23)
        for i in range(4):
            bits[20 + i] = (self.user_bits[2] >> i) & 1

        # Seconds tens (bits 24-26): Seconds // 10 in BCD
        sec_tens = self.seconds // 10
        bits[24] = sec_tens & 1
        bits[25] = (sec_tens >> 1) & 1
        bits[26] = (sec_tens >> 2) & 1

        # Bit 27: Polarity correction bit (for non-25fps rates)
        # This bit ensures even number of 0s in the frame (including sync)
        # We'll calculate this after all other bits are set
        bits[27] = 0  # Placeholder

        # User bits field 4 (bits 28-31)
        for i in range(4):
            bits[28 + i] = (self.user_bits[3] >> i) & 1

        # Minutes units (bits 32-35): Minutes % 10 in BCD (LSB first)
        min_units = self.minutes % 10
        for i in range(4):
            bits[32 + i] = (min_units >> i) & 1

        # User bits field 5 (bits 36-39)
        for i in range(4):
            bits[36 + i] = (self.user_bits[4] >> i) & 1

        # Minutes tens (bits 40-42): Minutes // 10 in BCD
        min_tens = self.minutes // 10
        bits[40] = min_tens & 1
        bits[41] = (min_tens >> 1) & 1
        bits[42] = (min_tens >> 2) & 1

        # Bit 43: Binary group flag (set to 0 for no format)
        bits[43] = 0

        # User bits field 6 (bits 44-47)
        for i in range(4):
            bits[44 + i] = (self.user_bits[5] >> i) & 1

        # Hours encoding (bits 48-57)
        # Bits 48-51: units digit (hours % 10), LSB first
        # Bits 52-57: tens digit (hours // 10), can be 0-102
        #   - Bits 52-55: upper bits of tens (values >= 4 indicate extended mode)
        #   - Bits 56-57: lower 2 bits of tens (SMPTE BCD position, LSB first)
        #
        # For SMPTE compatibility (hours 0-39):
        #   - Bits 48-51: BCD units (0-9)
        #   - Bits 56-57: BCD tens (0-3)
        #   - Bits 52-55: 0000 (indicates standard SMPTE)
        #
        # For extended mode (hours 40-1023):
        #   - Bits 52-55: non-zero (upper bits of tens value)
        #   - Bits 56-57: lower 2 bits of tens value
        #   - Standard SMPTE decoders read bits 56-57 only, showing incorrect values
        hour_units = self.hours % 10
        hour_tens = self.hours // 10

        # Bits 48-51: units (LSB first)
        for i in range(4):
            bits[48 + i] = (hour_units >> i) & 1

        # Bits 52-55: upper bits of tens (LSB first, bits 2-5 of tens value)
        for i in range(4):
            bits[52 + i] = ((hour_tens >> (i + 2)) & 1)

        # Bits 56-57: lower 2 bits of tens (SMPTE BCD position, LSB first)
        bits[56] = hour_tens & 1
        bits[57] = (hour_tens >> 1) & 1

        # Bit 58: Clock flag / Binary group flag 1 (set to 0 for arbitrary time origin)
        bits[58] = 0

        # Bit 59: Binary group flag (BGF) - set to 0 for no format
        bits[59] = 0

        # User bits field 8 (bits 60-63)
        # Bit 60 is used for FlexTC direction flag, bits 61-63 are user bits
        # Bit 60 = 0 means count up (standard SMPTE/LTC default)
        # Bit 60 = 1 means count down (FlexTC countdown mode)
        bits[60] = 0 if self.count_up else 1
        for i in range(1, 4):
            bits[60 + i] = (self.user_bits[7] >> i) & 1

        # Sync word (bits 64-79): 0011 1111 1111 1101 (SMPTE standard)
        sync_word = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
        for i, bit in enumerate(sync_word):
            bits[64 + i] = bit

        # Calculate polarity correction bit (bit 27)
        # Ensures even number of 0 bits in the whole frame (including sync)
        zero_count = bits[:27].count(0) + bits[28:].count(0)
        # If zero_count is odd, set bit 27 to 1 to make it even
        bits[27] = 1 if (zero_count % 2 == 1) else 0

        return bits

    @classmethod
    def decode_80bit(cls, bits: list[int]) -> Optional['Timecode']:
        """
        Decode 80-bit SMPTE/LTC format to timecode.

        Args:
            bits: List of 80 bits (0 or 1)

        Returns:
            Timecode object, or None if invalid
        """
        if len(bits) != 80:
            return None

        # Verify sync pattern (bits 64-79)
        # SMPTE/LTC sync word variants (different encoders use different bit patterns)
        # Standard: 0011 1111 1111 1101 (0x3FFD in our bit ordering)
        # Alternative: 0011 1111 1111 1111 (0x3FFF - used by some encoders)
        # The key is the first two bits must be 0011
        sync_actual = bits[64:80]

        # Check for valid sync patterns
        # Pattern must start with 0011 and have mostly 1s
        valid_syncs = [
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],  # Standard
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Alternative (all 1s after 0011)
        ]

        if sync_actual not in valid_syncs:
            return None

        # Extract frame units (bits 0-3, LSB first)
        frame_units = sum(bits[i] << i for i in range(4))

        # Extract frame tens (bits 8-9)
        frame_tens = bits[8] | (bits[9] << 1)

        # Extract frame rate from bit 10 (drop frame) and bit 11 (color frame)
        # Based on reference file analysis:
        # bit10=0, bit11=0: Non-drop (24, 30, 29.97 NDF, 23.98) - timing distinguishes
        # bit10=0, bit11=1: 25 fps
        # bit10=1, bit11=0: Drop frame (29.97 DF, 30 DF)
        # bit10=1, bit11=1: (reserved)
        if bits[10] == 0 and bits[11] == 0:
            # Non-drop - default to FPS_30, could be 24, 30, 29.97 NDF, or 23.98
            # Decoders should use timing to distinguish, but we default to 30
            frame_rate = FrameRate.FPS_30
        elif bits[10] == 0 and bits[11] == 1:
            frame_rate = FrameRate.FPS_25
        elif bits[10] == 1 and bits[11] == 0:
            # Drop frame - could be 29.97 DF or 30 DF
            # Default to 29.97 DF for compatibility
            frame_rate = FrameRate.FPS_29_97_DROP
        else:
            frame_rate = FrameRate.FPS_30

        # Extract seconds units (bits 16-19, LSB first)
        sec_units = sum(bits[16 + i] << i for i in range(4))

        # Extract seconds tens (bits 24-26)
        sec_tens = bits[24] | (bits[25] << 1) | (bits[26] << 2)

        # Extract minutes units (bits 32-35, LSB first)
        min_units = sum(bits[32 + i] << i for i in range(4))

        # Extract minutes tens (bits 40-42)
        min_tens = bits[40] | (bits[41] << 1) | (bits[42] << 2)

        # Extract hours (bits 48-57)
        # Bits 48-51: units digit (hours % 10), LSB first
        # Bits 52-55: upper bits of tens (bits 2-5 of tens value), LSB first
        # Bits 56-57: lower 2 bits of tens (SMPTE BCD position), LSB first
        hour_units = sum(bits[48 + i] << i for i in range(4))
        hour_tens_lower = bits[56] | (bits[57] << 1)
        hour_tens_upper = sum(bits[52 + i] << (i + 2) for i in range(4))
        hour_tens = hour_tens_lower | hour_tens_upper
        hours = hour_tens * 10 + hour_units

        # Extract direction flag (bit 60 - FlexTC extension)
        # Bit 60 = 0 means count up (standard SMPTE/LTC default)
        # Bit 60 = 1 means count down (FlexTC countdown mode)
        count_up = bits[60] == 0

        # Extract user bits
        # Note: Field 7 (bits 52-55) is now used for extended hours encoding
        # Field 8 (bits 60-63) has bit 60 used for direction flag
        user_bits = []
        field_starts = [4, 12, 20, 28, 36, 44, 60]  # Skip field 7 (52-55)
        for field_idx in range(7):  # Only 7 fields now (field 7 is used for extended hours)
            field_start = field_starts[field_idx]
            if field_idx == 6:  # Field 8 (bits 60-63)
                # Bit 60 is direction flag, only use bits 61-63 for user data
                # Store in lower 3 bits of the field
                user_bits.append(
                    (bits[61] << 1) | (bits[62] << 2) | (bits[63] << 3)
                )
            else:
                user_bits.append(sum(bits[field_start + i] << i for i in range(4)))

        return cls(
            hours=hours,
            minutes=min_tens * 10 + min_units,
            seconds=sec_tens * 10 + sec_units,
            frames=frame_tens * 10 + frame_units,
            frame_rate=frame_rate,
            count_up=count_up,
            user_bits=user_bits
        )

    def __str__(self) -> str:
        """Format timecode as HH:MM:SS:FF (or HH:MM:SS;FF for drop-frame)."""
        direction = "+" if self.count_up else "-"
        separator = ";" if self.is_drop_frame else ":"
        return f"{direction}{self.hours:02d}:{self.minutes:02d}:{self.seconds:02d}{separator}{self.frames:02d}"


def generate_countdown(hours: int = 0, minutes: int = 0,
                      seconds: int = 0, frames: int = 0,
                      frame_rate: FrameRate = FrameRate.FPS_30,
                      start_hours: int = 0, start_minutes: int = 0,
                      start_seconds: int = 0, start_frames: int = 0) -> list[Timecode]:
    """
    Generate a sequence of timecode values for countdown.

    Args:
        hours: Duration in hours
        minutes: Duration in minutes
        seconds: Duration in seconds
        frames: Duration in frames
        frame_rate: Frame rate
        start_hours: Starting hours
        start_minutes: Starting minutes
        start_seconds: Starting seconds
        start_frames: Starting frames

    Returns:
        List of Timecode objects counting down from start time
    """
    # Calculate total frames as a duration (not absolute timecode)
    # Map frame rate enum to integer fps value for calculation
    if frame_rate == FrameRate.FPS_24:
        fps = 24
    elif frame_rate == FrameRate.FPS_25:
        fps = 25
    elif frame_rate == FrameRate.FPS_29_97_NDF:
        fps = 30  # Use 30 for frame count purposes (29.97 is just timing)
    elif frame_rate == FrameRate.FPS_30:
        fps = 30
    elif frame_rate == FrameRate.FPS_23_98:
        fps = 24  # Use 24 for frame count purposes (23.98 is just timing)
    elif frame_rate == FrameRate.FPS_29_97_DROP:
        fps = 30
    elif frame_rate == FrameRate.FPS_30_DROP:
        fps = 30
    else:
        fps = 30  # Default

    # Calculate duration
    total_frames = hours * 3600 * fps + minutes * 60 * fps + seconds * fps + frames

    # Calculate starting point
    # If no start time specified, start from the duration (count down to zero)
    if start_hours == 0 and start_minutes == 0 and start_seconds == 0 and start_frames == 0:
        start_total = total_frames
    else:
        start_total = start_hours * 3600 * fps + start_minutes * 60 * fps + start_seconds * fps + start_frames

    result = []

    for i in range(total_frames + 1):
        remaining_frames = start_total - i
        if remaining_frames >= 0:
            tc = Timecode.from_total_frames(remaining_frames, frame_rate, count_up=False)
            result.append(tc)

    return result


def generate_countup(hours: int = 0, minutes: int = 0,
                     seconds: int = 0, frames: int = 0,
                     frame_rate: FrameRate = FrameRate.FPS_30,
                     start_hours: int = 0, start_minutes: int = 0,
                     start_seconds: int = 0, start_frames: int = 0) -> list[Timecode]:
    """
    Generate a sequence of timecode values for count-up.

    Args:
        hours: Duration in hours
        minutes: Duration in minutes
        seconds: Duration in seconds
        frames: Duration in frames
        frame_rate: Frame rate
        start_hours: Starting hours
        start_minutes: Starting minutes
        start_seconds: Starting seconds
        start_frames: Starting frames

    Returns:
        List of Timecode objects counting up from start time
    """
    # Calculate total frames as a duration (not absolute timecode)
    # Map frame rate enum to integer fps value for calculation
    if frame_rate == FrameRate.FPS_24:
        fps = 24
    elif frame_rate == FrameRate.FPS_25:
        fps = 25
    elif frame_rate == FrameRate.FPS_29_97_NDF:
        fps = 30  # Use 30 for frame count purposes (29.97 is just timing)
    elif frame_rate == FrameRate.FPS_30:
        fps = 30
    elif frame_rate == FrameRate.FPS_23_98:
        fps = 24  # Use 24 for frame count purposes (23.98 is just timing)
    elif frame_rate == FrameRate.FPS_29_97_DROP:
        fps = 30
    elif frame_rate == FrameRate.FPS_30_DROP:
        fps = 30
    else:
        fps = 30  # Default

    # Calculate starting point
    start_total = start_hours * 3600 * fps + start_minutes * 60 * fps + start_seconds * fps + start_frames

    # Calculate duration
    total_frames = hours * 3600 * fps + minutes * 60 * fps + seconds * fps + frames
    result = []

    for i in range(total_frames + 1):
        current_frames = start_total + i
        tc = Timecode.from_total_frames(current_frames, frame_rate, count_up=True)
        result.append(tc)

    return result
