# FlexTC - SMPTE-Compatible, Flexible, Bi-Directional, Extended Timecode

A SMPTE/LTC backwards-compatible timecode system that supports both standard count-up timecode and countdown mode. Uses native Biphase-M (Manchester) encoding for compatibility with standard SMPTE equipment.

This package includes a Python-based encoder that can be used to create audio files with SMPTE-compliant timecode, as well as a Python-based decoder that can decode audio files or use a live input. Additionally, when decoding from a live input, timecode can be distributed as an OSC string for easy feedback into other control systems.

<a href="https://www.buymeacoffee.com/alokw" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


## Overview

FlexTC provides two modes of operation:

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

Bit 60 of the frame indicates the timecode direction (FlexTC extension):
- **Bit 60 = 0**: Counting up (standard SMPTE mode - default)
- **Bit 60 = 1**: Counting down (countdown mode)

Bit 60 is the first bit of user bits field 8, which is reserved for future use in standard SMPTE. This allows a single decoder to handle both standard SMPTE timecode and FlexTC countdown streams.

**Note:** When decoding from files, the direction is determined by comparing the first and last timecode values, not solely by bit 60. This ensures compatibility with standard LTC files that may have bit 60 set to arbitrary values.

### Beyond 24-Hour Encoding

The SMPTE specification uses only 2 bits for the "hours tens" digit (bits 56-57), which can represent values 0-3, allowing encoding of 0-39 hours in standard SMPTE.

FlexTC implements an **extended hours encoding** using bits 52-55 (user bits field 7) to support up to 639 hours while remaining fully compatible with standard SMPTE decoders for hours 0-39.

#### Encoding Scheme

| Hours Range | Encoding Mode | Bits 52-55 | Bits 56-57 | Standard Decoder Sees |
|-------------|---------------|------------|------------|----------------------|
| 0-39 | BCD (SMPTE compatible) | 0000 | 0-3 | Correct hours ✓ |
| 40-639 | Binary (FlexTC extended) | Non-zero | 0-3 (lower bits of tens) | Incorrect hours |

**How it works:**
- **Bits 48-51**: Units digit (0-9), same for all modes
- **Bits 52-55**: Upper bits of tens value (bits 2-5 of `hours // 10`)
- **Bits 56-57**: Lower 2 bits of tens value (bits 0-1 of `hours // 10`)

For hours 0-39, tens is 0-3, so bits 52-55 are all zero → SMPTE compatible.
For hours 40-639, tens is 4-63, so bits 52-55 are non-zero → Extended mode.

**Decoding formula:** `hours = tens × 10 + units` where `tens = bits[52-55] << 2 | bits[56-57]`

#### Compatibility

**Hours 0-39**: Fully compatible with all SMPTE decoders (bits 52-55 = 0000)

**Hours 40-639**: Require FlexTC-aware decoder (or another decoder that also looks at bits 52-57 for extended hours); standard SMPTE decoders see only the lower 2 bits of the tens value

## Usage

```bash
# Install in editable/development mode (recommended)
pip install -e .

# This installs the console commands:
#   flextc-encode  - Generate timecode audio
#   flextc-decode  - Read timecode from audio
```

### Console Commands

```bash
# Encode timecode (output filename auto-generated if -o not specified)
flextc-encode 5m                      # Generates: ltc_30fps_5m.wav
flextc-encode 5m --countdown          # Generates: count_30fps_5m.wav

# Or specify custom output
flextc-encode 5m -o my_file          # Creates: my_file.wav

# Decode from file
flextc-decode -i ltc_30fps_5m.wav

# Show help
flextc-encode --help
flextc-decode --help
```

### Encoder

```bash
# Basic usage - count-up (default)
flextc-encode 5m                      # 5 minutes, generates: ltc_30fps_5m.wav
flextc-encode 1:30                    # 1 min 30 sec, generates: ltc_30fps_1m30s.wav

# Countdown mode
flextc-encode 5m --countdown          # Generates: count_30fps_5m.wav

# Specify custom output
flextc-encode 5m -o my_file          # Creates: my_file.wav

# Frame rate options
flextc-encode 10m -r 25              # 25 fps (PAL)
flextc-encode 10m -r 23.98           # 23.98 fps (film/HD)
flextc-encode 10m -r 24              # 24 fps (film production)
flextc-encode 10m -r 29.97           # 29.97 fps non-drop (NTSC)
flextc-encode 10m -r 29.97 --drop-frame  # 29.97 fps drop-frame

# Start from specific timecode
flextc-encode 10s --start 1:00:00:00  # Start from 1 hour
flextc-encode 1m --start 15:30:00:00  # Start from 15:30:00:00

# Sample rate and amplitude
flextc-encode 5m -s 44100            # 44.1 kHz sample rate
flextc-encode 5m -a 0.5              # Lower amplitude (0.0 to 1.0)

# Waveform type
flextc-encode 5m                    # Sine waveform (default, broadcast-friendly)
flextc-encode 5m --square            # Square waveform (~1μs rise time)
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
flextc-decode

# Decode from specific device
flextc-decode -d 2

# Decode from specific channel (0=left, 1=right)
flextc-decode -d 2 -c 1

# Decode from file
flextc-decode -i output.wav

# List available audio devices
flextc-decode --list-devices

# Specify frame rate (default: auto-detect)
flextc-decode -r 30

# Verbose output with statistics
flextc-decode -v

# Broadcast timecode via OSC
flextc-decode -d 2 -c 1 --osc --osc-address 127.0.0.1
```

### Graphical Interface (GUI)

FlexTC includes an optional native desktop GUI built with PySide6 (Qt for Python). The GUI provides the same encoding and decoding functionality as the CLI tools with a professional dark-themed interface.

**Installing GUI support:**
```bash
# Install with GUI dependencies
pip install -e ".[gui]"

# Or install PySide6 separately
pip install PySide6
```

**Running the GUI:**
```bash
flextc-gui
```

**GUI Features:**
- **Encoder Tab**: Generate timecode files with visual controls for all parameters
  - Duration input (supports all formats: `5m`, `1:30`, `2h30m`, `7h6m5s4f`)
  - Optional start timecode
  - Frame rate selection (23.98, 24, 25, 29.97, 30 fps)
  - Drop-frame mode toggle
  - Countdown mode toggle
  - Sample rate and amplitude controls
  - Auto-generated or custom output filenames
  - Progress indication during encoding

- **Decoder Tab**: Analyze timecode files
  - File browser for WAV files
  - Auto-detection of frame rate and direction
  - Results table showing start/end timecodes, duration, and format info

**About Qt/PySide6:**
- PySide6 is the official Python binding for Qt (a mature C++ GUI framework)
- **No additional tools required** - you don't need Qt Creator or any separate IDE
- The GUI code is pure Python and imports your existing `encoder.py` and `decoder.py` modules directly
- Applications run natively on Windows and Mac with the platform's look and feel
- Packaging for distribution is done with Python tools (see below)

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
flextc-decode -d 2 -c 1 --osc --osc-address 127.0.0.1

# Broadcast to a specific IP and port
flextc-decode --osc --osc-address 192.168.1.100 --osc-port 9999
```

**OSC Options:**
- `--osc` - Enable OSC broadcasting
- `--osc-address` - Target IP address (default: 255.255.255.255 for broadcast)
- `--osc-port` - UDP port (default: 9988)

**OSC Paths:**
- `/flextc/ltc` - Sent when counting up (standard SMPTE mode)
- `/flextc/count` - Sent when counting down (countdown mode)

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

The following hardware and software have been tested with FlexTC-generated timecode:

| Hardware/Software | Notes |
|-------------------|-------|
| **FlexTC Decoder** | Reads bi-directional timecode up to hour 639 (using extended user bits as noted) |
| **TouchDesigner** | Reads bi-directional timecode up to hour 39 |
| **Horae** | Reads bi-directional timecode up to hour 39 |
| **Brainstorm SR-112 Distripalyzer** | Reads forward timecode up to hour 23 |
| **GrandMA3** | Reads bi-directional timecode up to hour 39 (hour 24-39 represented as 1.1 to 1.15) |
| **Horita TR-100 Reader** | Reads bi-directional timecode up to hour 23 |

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

### Waveform Types

The encoder supports two waveform types for the LTC audio signal:

| Waveform | Description | Rise Time | Use Case |
|----------|-------------|-----------|----------|
| **Sine** | Smooth sinusoidal transitions | ~25μs | Default; broadcast-friendly, reduces harmonics |
| **Square** | Instant transitions between levels | ~1μs | Traditional for hardware LTC generators |

**Sine Wave** (default):
- Smoothed transitions reduce high-frequency harmonics
- Better suited for broadcast transmission and analog systems
- Maintains identical zero-crossing timing for decoder compatibility

**Square Wave** (`--square` option):
- Produces a clean digital signal with sharp transitions
- Ideal for direct connection to equipment expecting digital LTC
- Contains more harmonic content due to sharp edges

Both waveforms produce identical timecode data - the only difference is the spectral content of the audio signal. Modern LTC decoders work reliably with either waveform type.

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
flextc-decode -v
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
