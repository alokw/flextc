# TTRRTT - SMPTE-Compatible Bidirectional Timecode

A SMPTE/LTC-compatible timecode system that supports both standard count-up timecode and countdown mode. Uses native Biphase-M (Manchester) encoding for compatibility with standard SMPTE equipment.

This package includes a Python-based encoder that can be used to create audio files with SMPTE-compliant timecode, as well as a Python-based decoder that can decode audio files or use a live input. Additionally, when decoding from a live input, timecode can be distributed as an OSC string for easy feedback into other control systems.

<a href="https://www.buymeacoffee.com/alokw" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


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

Bit 60 is the first bit of user bits field 8, which is reserved for future use in standard SMPTE. This allows a single decoder to handle both standard SMPTE timecode and TTRRTT countdown streams.

**Note:** When decoding from files, the direction is determined by comparing the first and last timecode values, not solely by bit 60. This ensures compatibility with standard LTC files that may have bit 60 set to arbitrary values.

### Beyond 24-Hour Encoding

While the SMPTE specification includes a binary group flag at bit 27 that indicates "timecode exceeds 24 hours" (using a special 24-bit format), this encoder does not set that flag. Instead, it continues encoding linearly beyond 24 hours using the standard format. These files can be read by decoders that ignore or do not check the 24-hour flag, including this decoder and many hardware devices.

## Installation

```bash
# Install in editable/development mode (recommended)
pip install -e .

# This installs the console commands:
#   ttrrtt-encode  - Generate timecode audio
#   ttrrtt-decode  - Read timecode from audio
```

## Usage

### Console Commands

```bash
# Encode timecode (output filename auto-generated if -o not specified)
ttrrtt-encode 5m                      # Generates: ltc_30fps_5m.wav
ttrrtt-encode 5m --countdown          # Generates: count_30fps_5m.wav

# Or specify custom output
ttrrtt-encode 5m -o my_file          # Creates: my_file.wav

# Decode from file
ttrrtt-decode -i ltc_30fps_5m.wav

# Show help
ttrrtt-encode --help
ttrrtt-decode --help
```

### Encoder

```bash
# Basic usage - count-up (default)
ttrrtt-encode 5m                      # 5 minutes, generates: ltc_30fps_5m.wav
ttrrtt-encode 1:30                    # 1 min 30 sec, generates: ltc_30fps_1m30s.wav

# Countdown mode
ttrrtt-encode 5m --countdown          # Generates: count_30fps_5m.wav

# Specify custom output
ttrrtt-encode 5m -o my_file          # Creates: my_file.wav

# Frame rate options
ttrrtt-encode 10m -r 25              # 25 fps (PAL)
ttrrtt-encode 10m -r 23.98           # 23.98 fps (film/HD)
ttrrtt-encode 10m -r 24              # 24 fps (film production)
ttrrtt-encode 10m -r 29.97           # 29.97 fps non-drop (NTSC)
ttrrtt-encode 10m -r 29.97 --drop-frame  # 29.97 fps drop-frame

# Start from specific timecode
ttrrtt-encode 10s --start 1:00:00:00  # Start from 1 hour
ttrrtt-encode 1m --start 15:30:00:00  # Start from 15:30:00:00

# Sample rate and amplitude
ttrrtt-encode 5m -s 44100            # 44.1 kHz sample rate
ttrrtt-encode 5m -a 0.5              # Lower amplitude (0.0 to 1.0)
```

**Timecode Duration Formats:**
- `10s` = 10 seconds
- `5m` = 5 minutes
- `1h` = 1 hour
- `1:30` = 1 minute 30 seconds
- `1:30:00` = 1 minute 30 seconds
- `1:30:00:15` = 1 min 30 sec 15 frames
- `2h30m` = 2 hours 30 minutes (compound format)
- `7h6m5s4f` = 7 hours 6 minutes 5 seconds 4 frames (compound format)

**Timecode Start Formats (--start):**
- `1:00:00:00` = 1 hour
- `15:30:00:00` = 15 hours 30 minutes
- `1h` = 1 hour (also supports compound h/m/s/f notation)

**Auto-generated filename format:**
- Count-up: `ltc_{rate}fps{_drop}_{start}_{duration}.wav`
- Countdown: `count_{rate}fps{_drop}_{start}_{duration}.wav`
- `{rate}` = `30fps`, `2997fps`, `2398fps`, `24fps`, `25fps`
- `{_drop}` = optional suffix for drop-frame mode
- `{start}` = optional start timecode (HHMMSSFF format)
- `{duration}` = duration (e.g., `5m`, `30s`, `1h`, `2h30m`)

### Decoder

```bash
# Decode from default audio input (live)
ttrrtt-decode

# Decode from specific device
ttrrtt-decode -d 2

# Decode from specific channel (0=left, 1=right)
ttrrtt-decode -d 2 -c 1

# Decode from file
ttrrtt-decode -i output.wav

# List available audio devices
ttrrtt-decode --list-devices

# Specify frame rate (default: auto-detect)
ttrrtt-decode -r 30

# Verbose output with statistics
ttrrtt-decode -v

# Broadcast timecode via OSC
ttrrtt-decode -d 2 -c 1 --osc --osc-address 127.0.0.1
```

**File decoding output:**
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

### Display Indicators

The decoder displays direction with symbols:
- **▲** = Counting up (standard SMPTE)
- **▼** = Counting down (countdown mode)

Drop-frame timecodes use a semicolon separator before the frame number per SMPTE convention:

```
▼ 00:04:23;15  (packets: 3842)    # Countdown mode (drop-frame)
▲ 01:23:45:12  (packets: 5021)    # Count-up mode (non-drop)
```

**Separator difference:**
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

## Frame Rates

| Frame Rate | Option | Description | Drop-Frame Support |
|------------|--------|-------------|-------------------|
| 23.98 fps | `-r 23.98` | Film (24 × 1000/1001), HD video | No |
| 24 fps | `-r 24` | Film production | No |
| 25 fps | `-r 25` | PAL video | No |
| 29.97 fps | `-r 29.97` | NTSC timecode | Optional (`--drop-frame`) |
| 30 fps | `-r 30` | Standard audio/video (default) | Optional (`--drop-frame`) |

### Drop-Frame Mode

**Drop-frame mode** (`--drop-frame` flag) compensates for the difference between nominal (30) and actual (29.97) frame rate. It skips frame numbers 0 and 1 at the start of every minute except multiples of 10 minutes. Use for NTSC video to match real-time clock.

**Note:** Most non-drop frame rates (23.98, 24, 30, 29.97 NDF) encode with bits 10-11 = `00`. Decoders distinguish them by measuring the actual bit timing from the audio sample rate.

## Tested With

The following hardware and software have been tested with TTRRTT-generated timecode:

| Hardware/Software | Notes |
|-------------------|-------|
| **Horita Timecode Reader** | Reads bidirectional timecode up to 24 hours |
| **Brainstorm Distripalyzer** | Reads forward timecode up to 24 hours |
| **Horae** | All formats supported |
| **TouchDesigner** | All formats supported |

If you've tested with other equipment, please submit a PR to add it to this list.

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

### Robustness Features

1. **Sync word detection** - Reliable frame identification
2. **Error tolerance** - Up to 2 bit errors in sync word tolerated
3. **Continuous streaming** - Real-time processing with buffer
4. **Auto frame rate detection** - From bits 10-11 and timing analysis
5. **Direction auto-detection** - From bit 60 and timecode comparison
6. **Fractional timing support** - Accurate 29.97/23.98 fps encoding using sample accumulation
7. **Pause/resume resilience** - Fast recovery from audio interruptions with automatic decoder reset
8. **Corruption recovery** - Detects and recovers from biphase decoder misalignment

### Decoder State Machine

The decoder uses a two-state approach with automatic recovery mechanisms:

**Detection Mode**
- Entered on startup or when signal is lost
- Attempts initial lock with just 0.05 seconds of audio (fast detection)
- Accumulates progressively more data if initial lock fails
- Tries multiple buffer alignments when standard detection fails
- Uses bits 10 (drop frame) and 11 (color frame) to accurately identify frame rate
- Clears buffer when signal returns after silence
- Exits when valid timecodes are found

**Locked Mode**
- Decoder has frame rate and is processing timecodes continuously
- **200ms signal loss timeout** for fast recovery from interruptions
- Biphase decoder maintains state across callbacks for proper frame boundary handling
- **Automatic decoder reset** on detection of corruption or misalignment

**Corruption Recovery**

The decoder automatically detects and recovers from several corruption scenarios:
1. **Stuck Frames**: When the same timecode repeats 5+ times without progress
2. **Invalid Frames**: When frames are produced but no valid timecodes are decoded (3 consecutive callbacks)
3. **No Frames**: When strong signal produces no decoded frames (3 consecutive callbacks)

In all cases, the biphase decoder is reset while preserving frame rate knowledge, allowing fast re-synchronization without entering detection mode.

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
- python-osc (for OSC broadcasting)

## License

MIT

## Contributing

Contributions welcome! Please open issues or PRs.
