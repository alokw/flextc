# TTRRTT - SMPTE-Compatible Bidirectional Timecode

A SMPTE/LTC-compatible timecode system that supports both standard count-up timecode and countdown mode. Uses native Biphase-M (Manchester) encoding for compatibility with standard SMPTE equipment.

## Overview

TTRRTT provides two modes of operation:

1. **Countdown Mode**: Timecode counts down to zero - useful for timers, countdowns, remaining time display
2. **Count-Up Mode (Standard SMPTE)**: Traditional SMPTE timecode counting up from a specific start time

Both modes produce valid SMPTE/LTC audio that can be read by standard decoders. The direction is indicated by bit 60 of the frame.

## System Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  SMPTE Encoder  │──WAV───▶│  Audio Player  │──XLR───▶│  SMPTE Decoder  │
│                 │         │                │         │                 │
│ Bidirectional   │         │ Playback device│         │ Auto-detects    │
│ timecode gen    │         │ (any system)   │         │ direction       │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

## Specification

### Physical Layer
- **Encoding**: Biphase-M (Manchester) per SMPTE 12M / EBU Tech 3185
- **Frame rate**: 24, 25, 29.97 (drop-frame), or 30 fps
- **Sample rate**: 48 kHz (default) or 44.1 kHz
- **Bit rate**: 80 × frame_rate (e.g., 2400 bits/sec at 30 fps)

### Frame Structure
Each 80-bit SMPTE/LTC frame contains:

| Bits | Field | Description |
|------|-------|-------------|
| 0-3 | Frame units | Frames % 10 (BCD) |
| 4-7 | User bits field 1 | User data |
| 8-9 | Frame tens | Frames // 10 (BCD) |
| 10 | Drop frame flag | 1 = 29.97 df, 0 = non-drop |
| 11 | Color frame flag | 1 = 25 fps |
| 16-19 | Seconds units | Seconds % 10 (BCD) |
| 24-26 | Seconds tens | Seconds // 10 (BCD) |
| 32-35 | Minutes units | Minutes % 10 (BCD) |
| 40-42 | Minutes tens | Minutes // 10 (BCD) |
| 48-51 | Hours units | Hours % 10 (BCD) |
| 56-57 | Hours tens | Hours // 10 (BCD) |
| 58 | Clock flag | External clock sync |
| 59 | BGF / Polarity | Binary group flag / polarity correction |
| 60 | **Direction flag** | **0=count-up, 1=countdown** |
| 61-63 | User bits field 8 (partial) | User data |
| 64-79 | Sync word | Fixed: 0011 1111 1111 1101 |

### Direction Indicator

Bit 60 of the frame indicates the timecode direction (TTRRTT extension):
- **Bit 60 = 0**: Counting up (standard SMPTE mode - default)
- **Bit 60 = 1**: Counting down (countdown mode)

Bit 60 is the first bit of user bits field 8, which is reserved for future use in standard SMPTE. This avoids conflicts with the polarity correction bit (27) and binary group flags (43, 59) that are part of the SMPTE specification.

This allows a single decoder to handle both standard SMPTE timecode and TTRRTT countdown streams.

**Note:** When decoding from files, the direction is determined by comparing the first and last timecode values, not solely by bit 60. This ensures compatibility with standard LTC files that may have bit 60 set to arbitrary values.

## Installation

```bash
# Install in editable/development mode (recommended)
pip install -e .

# This installs the console commands:
#   ttrrtt-encode  - Generate timecode audio
#   ttrrtt-decode  - Read timecode from audio
```

## Usage

### Console Commands (Recommended)

The package installs two console commands that avoid Python module warnings:

```bash
# Encode timecode (output filename auto-generated if -o not specified)
ttrrtt-encode 5m                      # Generates: ltc_30fps_5m.wav

# Encode countdown
ttrrtt-encode 5m --countdown          # Generates: ltc_30fps_countdown_5m.wav

# Or specify custom output
ttrrtt-encode 5m -o my_file          # Creates: my_file.wav

# Decode from file
ttrrtt-decode -i ltc_30fps_5m.wav

# Show help
ttrrtt-encode --help
ttrrtt-decode --help
```

### Python Module Usage

You can also use the encoder/decoder as Python modules (may produce warnings when run from the package directory):

```bash
python -m ttrrtt.encoder 5m -o output.wav
python -m ttrrtt.decoder -i output.wav
```

### Frame Rates

The encoder supports the following frame rates:

| Frame Rate | Option | Description |
|------------|--------|-------------|
| 23.98 fps | `-r 23.98` | Film (24 * 1000/1001), HD video |
| 24 fps | `-r 24` | Film production |
| 25 fps | `-r 25` | PAL video |
| 29.97 fps | `-r 29.97` | NTSC timecode |
| 30 fps | `-r 30` | Standard audio/video (default) |

**Default**: 30 fps non-drop at 48 kHz sample rate

#### Drop-Frame vs Non-Drop

**Drop-frame mode** (`--drop-frame` flag):
- Compensates for the difference between nominal (30) and actual (29.97) frame rate
- Skips frame numbers 0 and 1 at the start of every minute except multiples of 10 minutes
- Use for NTSC video to match real-time clock

**29.97 non-drop** (`-r 29.97`):
- Encodes with bit 10 = 0 (no drop-frame flag)
- Slight drift from real-time clock over time
- This is the default when using `-r 29.97` without `--drop-frame`

**29.97 drop-frame** (`-r 29.97 --drop-frame`):
- NTSC drop-frame timecode
- Encodes with bit 10 = 1 (drop-frame flag set)
- Matches wall-clock time

**30 fps drop-frame** (`-r 30 --drop-frame`):
- Uses drop-frame encoding at nominal 30 fps

**30 fps non-drop** (`-r 30`):
- Standard for audio/video post-production
- Default mode when no frame rate is specified

**23.98 fps non-drop** (`-r 23.98`):
- Film transfer rate (24 * 1000/1001)
- Encodes with bit 10 = 0 (no drop-frame flag)

### Encoder (Generate Timecode Audio)

```bash
# Output filename is auto-generated if -o is not specified
# Format: ltc_{rate}fps{_drop}_{HHMMSSFF}_{duration}{_countdown}.wav
ttrrtt-encode 5m                              # Generates: ltc_30fps_5m.wav
ttrrtt-encode 10s --start 1:00:00:00          # Generates: ltc_30fps_01000000_10s.wav
ttrrtt-encode 1m -r 29.97 --drop-frame        # Generates: ltc_2997fps_drop_1m.wav
ttrrtt-encode 30s --countdown                 # Generates: ltc_30fps_countdown_30s.wav
ttrrtt-encode 1m -r 23.98                     # Generates: ltc_2398fps_1m.wav

# Specify custom output with -o
ttrrtt-encode 5m -o my_file                  # Creates: my_file.wav (.wav auto-added)

# Specify frame rate (23.98, 24, 25, 29.97, or 30)
ttrrtt-encode 10m -r 25 -o output_25fps
ttrrtt-encode 10m -r 23.98 -o output_2398fps
ttrrtt-encode 10m -r 24 -o output_24fps

# 29.97 non-drop (default for -r 29.97)
ttrrtt-encode 10m -r 29.97 -o output_2997ndf

# 29.97 drop-frame
ttrrtt-encode 10m -r 29.97 --drop-frame -o output_2997df

# 30 drop-frame
ttrrtt-encode 10m -r 30 --drop-frame -o output_30df

# Countdown mode
ttrrtt-encode 5m --countdown -o countdown_5min

# Start from specific timecode
ttrrtt-encode 10s --start 1:00:00:00 -o from_1hr
ttrrtt-encode 1m --start 15:30:00:00 -o from_15_30

# Specify sample rate (default: 48000 Hz)
ttrrtt-encode 5m -s 44100 -o output_44k

# Adjust amplitude (0.0 to 1.0, default: 0.7)
ttrrtt-encode 5m -a 0.5 -o output_quiet
```

**Auto-generated filename format:**
- `ltc_` - prefix
- `{rate}fps` - frame rate (e.g., `30fps`, `2997fps`, `2398fps`, `24fps`, `25fps`)
- `{_drop}` - optional suffix for drop-frame mode
- `{_HHMMSSFF}` - optional start timecode (no "_from" prefix)
- `{_countdown}` - optional suffix for countdown mode
- `_{duration}` - duration (e.g., `5m`, `30s`, `1h`, `1h30m`)
- `.wav` - extension

**Timecode formats:**
- `10s` = 10 seconds
- `5m` = 5 minutes
- `1h` = 1 hour
- `1:30` = 1 minute 30 seconds
- `1:30:00` = 1 minute 30 seconds
- `1:30:00:15` = 1 min 30 sec 15 frames

### Decoder (Read Timecode from Audio)

```bash
# Decode from default audio input (live)
ttrrtt-decode

# Decode from specific device
ttrrtt-decode -d 2

# Decode from specific channel (0=left, 1=right)
ttrrtt-decode -d 2 -c 1

# Decode from file
# When decoding from a file, the decoder reads the entire file to determine:
# - Start timecode (first valid frame)
# - End timecode (last valid frame)
# - Duration (calculated from timecode values)
# - Direction (determined by comparing start/end timecodes)
# - Frame rate (auto-detected)
ttrrtt-decode -i output.wav

# List available audio devices
ttrrtt-decode --list-devices

# Specify frame rate
ttrrtt-decode -r 30

# Verbose output with statistics
ttrrtt-decode -v

# Broadcast timecode via OSC
ttrrtt-decode -d 2 -c 1 --osc --osc-address 127.0.0.1
```

**File decoding output example:**
```
Decoding from file: ltc_2997fps_drop_30s.wav
----------------------------------------
Auto-detected frame rate: 29.97 fps

Start timecode: ▲ 00:00:00;00
End timecode:   ▲ 00:00:29;26
Duration:       00:00:29;26
Direction:      Counting up
Frame rate:     29.97 fps
```

### Example Workflow

```bash
# 1. Generate a 10-second count-up test file
ttrrtt-encode 10s -o test_countup -v

# 2. Verify it decodes correctly
ttrrtt-decode -i test_countup.wav

# 3. Generate a countdown test file
ttrrtt-encode 10s --countdown -o test_countdown -v

# 4. Verify it decodes as countdown
ttrrtt-decode -i test_countdown.wav
```

### Display Indicators

The decoder displays direction with symbols:
- **▲** = Counting up (standard SMPTE)
- **▼** = Counting down (countdown mode)

Drop-frame timecodes use a semicolon separator before the frame number per SMPTE convention:

Example output:
```
▼ 00:04:23;15  (packets: 3842)    # Countdown mode (drop-frame)
▲ 01:23:45:12  (packets: 5021)    # Count-up mode (non-drop)
```

**Note the separator difference:**
- `:` (colon) = Non-drop frame timecode (HH:MM:SS:FF)
- `;` (semicolon) = Drop-frame timecode (HH:MM:SS;FF)

### OSC Broadcasting

The decoder can broadcast decoded timecode via OSC (Open Sound Control) for integration with other systems. This allows software like show control systems, DAWs, or custom applications to receive and display the timecode.

**Important:** OSC broadcasts are sent periodically (approximately every 100ms), **not on every frame**. This feature is intended for display and reference purposes only and should **not** be used for frame-accurate triggering or synchronization.

```bash
# Broadcast timecode via OSC to localhost
ttrrtt-decode -d 2 -c 1 --osc --osc-address 127.0.0.1

# Broadcast to a specific IP and port
ttrrtt-decode --osc --osc-address 192.168.1.100 --osc-port 9999
```

**OSC Options:**
- `--osc` - Enable OSC broadcasting
- `--osc-address` - Target IP address (default: 255.255.255.255 for broadcast)
- `--osc-port` - UDP port (default: 9988)

**OSC Paths:**
- `/ttrrtt/ltc` - Sent when counting up (standard SMPTE mode)
- `/ttrrtt/count` - Sent when counting down (countdown mode)

**Message Format:** Single string argument in format `HH:MM:SS:FF` or `HH:MM:SS;FF` (for drop-frame).

## Compatibility

### Standard SMPTE Equipment
- Generated files are valid SMPTE/LTC audio
- Can be read by standard SMPTE decoders (will show as count-up)
- Direction flag (bit 60) is ignored by standard equipment

### Frame Rates
- **23.98 fps**: Film transfer rate, HD video (use `-r 23.98`)
- **24 fps**: Film production (use `-r 24`)
- **25 fps**: PAL video (use `-r 25`)
- **29.97 fps**: NTSC non-drop by default, use `--drop-frame` for drop-frame
- **30 fps**: Standard audio/video (default, use `-r 30`)

**Drop-frame variants**:
- 29.97 non-drop: `-r 29.97` (default)
- 29.97 drop-frame: `-r 29.97 --drop-frame`
- 30 drop-frame: `-r 30 --drop-frame`

**Note**: Most non-drop frame rates (23.98, 24, 30, 29.97 NDF) encode with bits 10-11 = `00`. Decoders distinguish them by measuring the actual bit timing from the audio sample rate.

## Technical Details

### Biphase-M Encoding Rules
1. Every bit cell starts with a transition
2. Logic 0: Additional transition in middle of cell
3. Logic 1: No transition in middle

This ensures:
- Guaranteed clock recovery (minimum one transition per bit)
- DC-free encoding (balanced signal)
- Robust to polarity inversion

### Decoder Algorithm
1. Detect edges (zero-crossings) in audio signal
2. Measure edge-to-edge distances
3. Distance ≈ half period → logic 0
4. Distance ≈ full period → logic 1
5. Verify sync word (bits 64-79)
6. Extract timecode and direction flag

## Comparison to Standard SMPTE

| Feature | Standard SMPTE | TTRRTT |
|---------|----------------|--------|
| Direction | Forward only | Forward **or** Reverse |
| Frame rates | 24/25/29.97/30 | 23.98/24/25/29.97/30 |
| Encoding | Biphase-M | Biphase-M |
| Duration | ~24 hours | ~24 hours |
| Bit 60 | User bit (typically 0) | **Direction indicator** (0=count-up, 1=countdown) |
| Compatibility | Universal | Compatible with standard decoders |

## Robustness Features

1. **Sync word detection** - Reliable frame identification
2. **Error tolerance** - Up to 2 bit errors in sync word tolerated
3. **Continuous streaming** - Real-time processing with buffer
4. **Auto frame rate detection** - From bits 10-11 and timing analysis
5. **Direction auto-detection** - From bit 60
6. **Fractional timing support** - Accurate 29.97/23.98 fps encoding using sample accumulation
7. **Pause/resume resilience** - Fast recovery from audio interruptions with automatic decoder reset
8. **Corruption recovery** - Detects and recovers from biphase decoder misalignment

## Decoder State Machine

The decoder uses a two-state approach with automatic recovery mechanisms:

### Detection Mode
- Entered on startup or when signal is lost
- Attempts initial lock with just 0.05 seconds of audio (fast detection)
- Accumulates progressively more data if initial lock fails
- Tries multiple buffer alignments when standard detection fails
- Uses bits 10 (drop frame) and 11 (color frame) to accurately identify frame rate
- Clears buffer when signal returns after silence
- Exits when valid timecodes are found

### Locked Mode
- Decoder has frame rate and is processing timecodes continuously
- **200ms signal loss timeout** for fast recovery from interruptions
- Biphase decoder maintains state across callbacks for proper frame boundary handling
- **Automatic decoder reset** on detection of corruption or misalignment

### Corruption Recovery

The decoder automatically detects and recovers from several corruption scenarios:

1. **Stuck Frames**: When the same timecode repeats 5+ times without progress
2. **Invalid Frames**: When frames are produced but no valid timecodes are decoded (3 consecutive callbacks)
3. **No Frames**: When strong signal produces no decoded frames (3 consecutive callbacks)

In all cases, the biphase decoder is reset while preserving frame rate knowledge, allowing fast re-synchronization without entering detection mode.

### Pause/Resume Behavior

When audio is interrupted (pause, cable disconnect, etc.):

1. **Signal Loss** (200ms timeout):
   - Decoder enters detection mode
   - Biphase decoder is discarded
   - Waiting for signal... message displayed

2. **Signal Returns**:
   - Detection buffer is cleared of stale data
   - Frame rate is re-detected
   - Decoder locks onto signal and resumes tracking

3. **Fast Pause/Resume** (< 200ms):
   - Decoder remains in locked mode
   - Biphase decoder is automatically reset if corrupted
   - Minimal disruption to timecode tracking

The biphase decoder's internal buffers are preserved during normal operation to ensure frame boundaries are correctly handled across audio callbacks. This allows seamless decoding even when frame boundaries span multiple callbacks.

### Signal Transition Handling
- **Silence → Signal**: Detection buffer is cleared when signal returns after silence
- **Signal → Silence**: Samples with signal < 0.01 are skipped entirely
- **Pause/Resume**: Decoder automatically recovers when audio signal resumes after brief interruptions

## Debug Logging

Enable verbose logging to see state transitions:

```bash
ttrrtt-decode -v
```

Key log messages:
- `[SIGNAL LOSS]` - Entered detection mode (200ms timeout)
- `[SIGNAL RETURN]` - Signal detected after silence
- `[DETECTION]` - Frame rate detection messages
- `[STUCK FRAMES]` - Timecode stuck repeating, resetting biphase decoder
- `[INVALID FRAMES]` - Frames produced but no valid timecodes, resetting biphase decoder
- `[NO FRAMES]` - Strong signal but no decoded frames, resetting biphase decoder

## Requirements

- Python 3.9+
- numpy
- soundfile
- sounddevice
- scipy (for file resampling in decoder)

## License

MIT

## Contributing

Contributions welcome! Please open issues or PRs.
