"""
FlexTC - SMPTE-Compatible Bidirectional Timecode
A SMPTE/LTC-compatible timecode system with countdown support.
"""

__version__ = "0.1.0"

from .smpte_packet import Timecode, FrameRate, generate_countdown, generate_countup
from .encoder import Encoder
from .decoder import Decoder

# Aliases for backwards compatibility
SMPTEEncoder = Encoder
SMPTEDecoder = Decoder

__all__ = [
    "Timecode",
    "FrameRate",
    "generate_countdown",
    "generate_countup",
    "Encoder",
    "Decoder",
    "SMPTEEncoder",  # Alias
    "SMPTEDecoder",  # Alias
]
