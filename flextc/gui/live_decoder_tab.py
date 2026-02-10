"""
Live Decoder Tab - GUI interface for real-time timecode decoding from audio input

This tab provides a graphical interface for live decoding of SMPTE/LTC timecode
from audio input devices. It uses the existing Decoder class from decoder.py.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QComboBox, QSpinBox, QPushButton, QLabel, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

import sounddevice as sd

from flextc.decoder import Decoder, format_timecode
from flextc.smpte_packet import Timecode


class LiveDecoderTab(QWidget):
    """
    Live decoder tab widget.

    Provides a GUI for real-time decoding of timecode from audio input.
    Features device selection, channel selection, and live timecode display.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.decoder: Optional[Decoder] = None
        self.is_running = False
        self._setup_ui()
        self._populate_audio_devices()

        # Timer for updating the timecode display
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.setInterval(100)  # Update 10 times per second

    def _setup_ui(self):
        """Set up the live decoder UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === Audio Device Selection Section ===
        device_group = QGroupBox("Audio Input Device")
        device_layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.setMinimumWidth(400)
        device_layout.addRow("Device:", self.device_combo)

        # Refresh button for device list
        refresh_button = QPushButton("Refresh Devices")
        refresh_button.clicked.connect(self._populate_audio_devices)
        device_layout.addRow("", refresh_button)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # === Settings Section ===
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()

        # Channel selection
        self.channel_combo = QComboBox()
        self.channel_combo.addItems([
            "Channel 1 (Left)",
            "Channel 2 (Right)",
        ])
        self.channel_combo.setCurrentIndex(0)
        settings_layout.addRow("Audio Channel:", self.channel_combo)

        # Sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            "48000 Hz (Default)",
            "44100 Hz",
        ])
        self.sample_rate_combo.setCurrentIndex(0)
        settings_layout.addRow("Sample Rate:", self.sample_rate_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === Start/Stop Button ===
        self.control_button = QPushButton("Start Decoding")
        self.control_button.setStyleSheet("""
            QPushButton {
                background-color: #2a82da;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a92ea;
            }
            QPushButton:pressed {
                background-color: #1a72ca;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
        """)
        self.control_button.clicked.connect(self._toggle_decoding)
        layout.addWidget(self.control_button)

        # === Timecode Display Section ===
        display_group = QGroupBox("Timecode Display")
        display_layout = QVBoxLayout()

        # Main timecode display - large font
        self.timecode_label = QLabel("--:--:--:--")
        self.timecode_label.setAlignment(Qt.AlignCenter)
        self.timecode_label.setFont(QFont("Monospace", 48, QFont.Bold))
        self.timecode_label.setStyleSheet("""
            QLabel {
                background-color: #000;
                color: #0f0;
                padding: 20px;
                border: 2px solid #444;
                border-radius: 5px;
            }
        """)
        display_layout.addWidget(self.timecode_label)

        # Status info (frame rate, packets, etc.)
        self.status_label = QLabel("Status: Stopped")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                color: #888;
                padding: 5px;
            }
        """)
        display_layout.addWidget(self.status_label)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # Add stretch to push everything up
        layout.addStretch()

    def _populate_audio_devices(self):
        """Populate the audio device dropdown with available input devices."""
        current_device = self.device_combo.currentData()
        self.device_combo.clear()

        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    # Add device name and index
                    name = f"[{i}] {dev['name']}"
                    self.device_combo.addItem(name, userData=i)
        except Exception as e:
            # Add error item
            self.device_combo.addItem(f"Error: {str(e)}", userData=None)

        # Try to restore previous selection
        if current_device is not None:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == current_device:
                    self.device_combo.setCurrentIndex(i)
                    break

    def _get_selected_device(self) -> Optional[int]:
        """Get the selected audio device index."""
        return self.device_combo.currentData()

    def _get_selected_channel(self) -> int:
        """Get the selected audio channel (0 or 1)."""
        return self.channel_combo.currentIndex()

    def _get_sample_rate(self) -> int:
        """Get sample rate from combo box."""
        text = self.sample_rate_combo.currentText()
        if "44100" in text:
            return 44100
        return 48000

    def _toggle_decoding(self):
        """Start or stop decoding."""
        if self.is_running:
            self._stop_decoding()
        else:
            self._start_decoding()

    def _start_decoding(self):
        """Start the decoding process."""
        device = self._get_selected_device()
        channel = self._get_selected_channel()
        sample_rate = self._get_sample_rate()

        # Create and start decoder
        try:
            self.decoder = Decoder(
                sample_rate=sample_rate,
                frame_rate=None,  # Auto-detect
                device=device,
                channel=channel,
                debug=False,
            )
            self.decoder.start()

            self.is_running = True
            self.control_button.setText("Stop Decoding")
            self.control_button.setStyleSheet("""
                QPushButton {
                    background-color: #d62a2a;
                    color: white;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #e63a3a;
                }
                QPushButton:pressed {
                    background-color: #c61a1a;
                }
            """)

            # Disable settings controls while running
            self.device_combo.setEnabled(False)
            self.channel_combo.setEnabled(False)
            self.sample_rate_combo.setEnabled(False)

            # Start the update timer
            self.update_timer.start()

            self.status_label.setText("Status: Detecting signal...")

        except Exception as e:
            self.status_label.setText(f"Error: {str(e)}")

    def _stop_decoding(self):
        """Stop the decoding process."""
        if self.decoder:
            self.decoder.stop()
            self.decoder = None

        self.is_running = False
        self.control_button.setText("Start Decoding")
        self.control_button.setStyleSheet("""
            QPushButton {
                background-color: #2a82da;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a92ea;
            }
            QPushButton:pressed {
                background-color: #1a72ca;
            }
        """)

        # Re-enable settings controls
        self.device_combo.setEnabled(True)
        self.channel_combo.setEnabled(True)
        self.sample_rate_combo.setEnabled(True)

        # Stop the update timer
        self.update_timer.stop()

        # Reset display
        self.timecode_label.setText("--:--:--:--")
        self.status_label.setText("Status: Stopped")

    def _update_display(self):
        """Update the timecode display with current data."""
        if not self.decoder or not self.is_running:
            return

        tc = self.decoder.get_timecode()
        stats = self.decoder.get_statistics()

        if tc:
            # Format timecode for display (without direction symbol for cleaner look)
            separator = ";" if tc.is_drop_frame else ":"
            tc_str = f"{tc.hours:02d}:{tc.minutes:02d}:{tc.seconds:02d}{separator}{tc.frames:02d}"
            self.timecode_label.setText(tc_str)

            # Update status
            direction = "▲" if tc.count_up else "▼"
            fps = tc.fps if tc.fps else "Unknown"
            self.status_label.setText(
                f"Status: Locked | {direction} {fps} fps | Frames: {stats['packets_received']}"
            )

            # Set green color when locked
            self.timecode_label.setStyleSheet("""
                QLabel {
                    background-color: #000;
                    color: #0f0;
                    padding: 20px;
                    border: 2px solid #444;
                    border-radius: 5px;
                }
            """)
        else:
            # Check if we're still detecting
            is_detecting = self.decoder._in_detection_mode or self.decoder.biphase is None

            if is_detecting:
                self.timecode_label.setText("DETECTING...")
                self.status_label.setText(
                    f"Status: Detecting frame rate... ({int(self.decoder._detection_buffer_samples / self.decoder.sample_rate * 100) / 100}s buffered)"
                )
                # Yellow color for detecting
                self.timecode_label.setStyleSheet("""
                    QLabel {
                        background-color: #000;
                        color: #ff0;
                        padding: 20px;
                        border: 2px solid #444;
                        border-radius: 5px;
                    }
                """)
            else:
                self.timecode_label.setText("--:--:--:--")
                self.status_label.setText("Status: Waiting for signal...")
                # Red color for no signal
                self.timecode_label.setStyleSheet("""
                    QLabel {
                        background-color: #000;
                        color: #f00;
                        padding: 20px;
                        border: 2px solid #444;
                        border-radius: 5px;
                    }
                """)

    def closeEvent(self, event):
        """Handle widget close event - stop decoding if running."""
        if self.is_running:
            self._stop_decoding()
        event.accept()
