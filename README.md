# RLTC - Reverse Linear Timecode

A robust FSK-based audio timecode system for countdown timing applications, inspired by LTC/SMPTE timecode but operating in reverse (counting down to zero).

## Overview

RLTC consists of two tools:

1. **Encoder (`rltc_encode`)**: Generates audio files containing countdown timecode data
2. **Decoder (`rltc_decode`)**: Real-time decodes RLTC from an audio input signal

## Use Case

Create audio files of specified lengths (10 seconds to 5 minutes typical, max 24 hours) that can be played back through any standard audio system. The analog output (XLR) carries a countdown timer that a decoder can read in real-time, even with moderate signal degradation.

## System Architecture

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   RLTC Encoder  │──WAV───▶│  Audio Player  │──XLR───▶│   RLTC Decoder  │
│                 │         │                │         │                 │
│ Generates file  │         │ Playback device│         │ Outputs time    │
│ with countdown  │         │ (any system)   │         │ remaining       │
└─────────────────┘         └─────────────────┘         └─────────────────┘
```

## Specification

### Physical Layer
- **Modulation**: BFSK (Binary Frequency Shift Keying)
- **Mark frequency**: 2200 Hz (logic 1)
- **Space frequency**: 1200 Hz (logic 0)
- **Baud rate**: 1200 baud
- **Sample rate**: 44.1 kHz or 48 kHz

These frequencies were chosen because:
- They work well on typical audio equipment (voice frequency range)
- They're distinguishable even with some noise/distortion
- Similar to Bell 202 modem standard (proven reliability)

### Packet Structure
Each packet contains:

| Field | Bits | Description |
|-------|------|-------------|
| Preamble | 32 | Alternating bits for sync (0xAAAAAAAA) |
| Sync Word | 16 | Fixed 0x16A3 for packet identification |
| Time Remaining | 32 | Milliseconds remaining (unsigned) |
| Packet Counter | 16 | Increments each packet |
| CRC-16 | 16 | Error detection (CRC-16-CCITT) |
| **Total** | **112** | ~9.33 ms per packet at 1200 baud |

### Transmission Rate
- **Standard**: 30 packets/second (33.3 ms interval)
- **Maximum practical**: ~100 packets/second (10 ms interval)
- **Minimum recommended**: 10 packets/second (100 ms interval)

At 30 Hz with a 33.3ms interval, the actual packet transmission takes ~9.3ms, leaving ~24ms of silence/guard time.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or for development
pip install -e .
```

## Usage

### Encoding

```bash
# Generate a 5-minute countdown file
rltc_encode --duration 300 --output countdown_5min.wav

# Generate a 30-second countdown at 25 fps
rltc_encode --duration 30 --fps 25 --output countdown_30s.wav
```

### Decoding

```bash
# Decode from default audio input
rltc_decode

# Decode from specific device
rltc_decode --device 2

# Decode from file (testing)
rltc_decode --input countdown_5min.wav
```

## Robustness Features

1. **CRC-16 error checking** - Detects corrupted packets
2. **Packet reassembly** - Uses packet counter to detect missing packets
3. **Moving average filtering** - Smooths out noisy readings
4. **Automatic gain normalization** - Handles varying signal levels
5. **Frequency hysteresis** - Reduces false bit detection from noise
6. **Preamble detection** - Reliable packet synchronization

## Limitations

- **Maximum duration**: 24 hours (32-bit millisecond counter)
- **Minimum duration**: ~100 ms (one packet)
- **Requires reliable audio path**: Analog XLR recommended for best results

## Comparison to LTC/SMPTE

| Feature | LTC/SMPTE | RLTC |
|---------|-----------|------|
| Direction | Forward (counting up) | Reverse (counting down) |
| Frame rate | 24/25/29.97/30 fps | Configurable (default 30 Hz) |
| Encoding | Biphase-M | BFSK |
| Max duration | ~24 hours | ~24 hours |
| Typical use | Video sync | Countdown timers |

## License

MIT

## Contributing

Contributions welcome! Please open issues or PRs.





Next Steps to Use

# Install dependencies
pip install -r requirements.txt

# Generate a 5-minute countdown file
python cli/encode.py 5m -o countdown.wav

# Test decode from the file
python cli/decode.py -i countdown.wav

# Decode live audio
python cli/decode.py

# List available devices
python cli/decode.py --list-devices