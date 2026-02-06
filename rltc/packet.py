"""
RLTC Packet structure and serialization.
"""

import struct
from typing import Optional


class CRC16CCITT:
    """
    CRC-16-CCITT (0x1021) implementation.
    Polynomial: x^16 + x^12 + x^5 + 1
    Initial value: 0xFFFF
    Final XOR: 0x0000
    """

    POLYNOMIAL = 0x1021
    INITIAL = 0xFFFF

    @classmethod
    def compute(cls, data: bytes) -> int:
        """Compute CRC-16-CCITT checksum."""
        crc = cls.INITIAL

        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = (crc << 1) ^ cls.POLYNOMIAL
                else:
                    crc <<= 1
                crc &= 0xFFFF

        return crc

    @classmethod
    def verify(cls, data: bytes, checksum: int) -> bool:
        """Verify data against CRC checksum."""
        return cls.compute(data) == checksum


class Packet:
    """
    Represents a single RLTC packet.

    Packet structure (MSB first):
    - Preamble: 32 bits (0xAAAAAAAA) - for synchronization
    - Sync Word: 16 bits (0x16A3) - identifies packet start
    - Time Remaining: 32 bits - milliseconds remaining (unsigned)
    - Packet Counter: 16 bits - increments each packet
    - CRC-16: 16 bits - error detection

    Total: 112 bits = 14 bytes
    """

    PREAMBLE = 0xAAAAAAAA  # 32 bits of alternating 1s and 0s
    SYNC_WORD = 0x16A3  # Chosen for good autocorrelation

    def __init__(
        self,
        time_remaining_ms: int,
        packet_counter: int,
        sync_word: int = SYNC_WORD,
    ):
        """
        Initialize a packet.

        Args:
            time_remaining_ms: Time remaining in milliseconds (0 to ~4 billion)
            packet_counter: Packet sequence number (0 to 65535, wraps)
            sync_word: Sync word (default 0x16A3)
        """
        if not 0 <= time_remaining_ms <= 0xFFFFFFFF:
            raise ValueError("time_remaining_ms must be 32-bit unsigned")
        if not 0 <= packet_counter <= 0xFFFF:
            raise ValueError("packet_counter must be 16-bit unsigned")

        self.time_remaining_ms = time_remaining_ms
        self.packet_counter = packet_counter
        self.sync_word = sync_word
        self.crc: Optional[int] = None

    def encode(self) -> bytes:
        """
        Encode packet to bytes (without preamble).

        Returns:
            10 bytes: sync_word (2) + time_remaining (4) + counter (2) + crc (2)
        """
        # Pack: sync (2), time (4), counter (2)
        payload = struct.pack(
            ">HIH",  # big-endian: unsigned short, unsigned int, unsigned short
            self.sync_word,
            self.time_remaining_ms,
            self.packet_counter,
        )

        # Compute CRC over payload
        self.crc = CRC16CCITT.compute(payload)

        # Return payload + CRC
        return payload + struct.pack(">H", self.crc)

    @classmethod
    def decode(cls, data: bytes) -> Optional["Packet"]:
        """
        Decode packet from bytes (without preamble).

        Args:
            data: 10 bytes containing sync, time, counter, and crc

        Returns:
            Packet if valid, None if CRC check fails
        """
        if len(data) != 10:
            return None

        # Unpack
        sync_word, time_remaining, counter, crc = struct.unpack(">HIHH", data)

        # Verify CRC
        payload = data[:8]  # sync + time + counter (no CRC)
        if not CRC16CCITT.verify(payload, crc):
            return None

        # Verify sync word
        if sync_word != cls.SYNC_WORD:
            return None

        return cls(time_remaining, counter, sync_word)

    def encode_with_preamble(self) -> bytes:
        """
        Encode full packet with preamble.

        Returns:
            16 bytes: preamble (4) + encoded packet (12)
        """
        preamble = struct.pack(">I", self.PREAMBLE)
        return preamble + self.encode()

    @classmethod
    def get_preamble_bits(cls) -> list[int]:
        """Get preamble as list of bits (MSB first)."""
        bits = []
        for i in range(31, -1, -1):
            bits.append((cls.PREAMBLE >> i) & 1)
        return bits

    @classmethod
    def get_sync_bits(cls) -> list[int]:
        """Get sync word as list of bits (MSB first)."""
        bits = []
        for i in range(15, -1, -1):
            bits.append((cls.SYNC_WORD >> i) & 1)
        return bits

    def __repr__(self) -> str:
        return (
            f"Packet(time={self.time_remaining_ms}ms, "
            f"counter={self.packet_counter}, crc={self.crc})"
        )
