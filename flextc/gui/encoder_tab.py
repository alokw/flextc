"""
Encoder Tab - GUI interface for timecode encoding

This tab provides a graphical interface for the encoder functionality.
It imports and uses the existing Encoder class from encoder.py directly.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QFileDialog, QProgressBar, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDoubleValidator

from flextc.encoder import Encoder, parse_timecode


class EncoderWorker(QThread):
    """
    Worker thread for encoding timecode files.

    Runs the encoding operation in a separate thread to keep
    the GUI responsive during long operations.
    """

    progress = Signal(int)  # Progress percentage (0-100)
    finished = Signal(str)  # Output file path on success
    error = Signal(str)     # Error message on failure

    def __init__(
        self,
        output_path: str,
        hours: int,
        minutes: int,
        seconds: int,
        frames: int,
        countdown: bool,
        start_hours: int,
        start_minutes: int,
        start_seconds: int,
        start_frames: int,
        sample_rate: int,
        frame_rate: float,
        amplitude: float,
        drop_frame: bool,
        waveform: str,
    ):
        super().__init__()
        self.output_path = output_path
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.frames = frames
        self.countdown = countdown
        self.start_hours = start_hours
        self.start_minutes = start_minutes
        self.start_seconds = start_seconds
        self.start_frames = start_frames
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.amplitude = amplitude
        self.drop_frame = drop_frame
        self.waveform = waveform
        self._cancelled = False

    def stop(self):
        """Cancel the encoding operation."""
        self._cancelled = True

    def run(self):
        """Run the encoding operation in the background thread."""
        try:
            # Check for cancellation before starting
            if self._cancelled:
                self.error.emit("Encoding cancelled")
                return

            # Create encoder using the existing class
            encoder = Encoder(
                sample_rate=self.sample_rate,
                frame_rate=self.frame_rate,
                amplitude=self.amplitude,
                drop_frame=self.drop_frame,
                waveform=self.waveform,
            )

            # Generate to file
            encoder.generate_to_file(
                output_path=self.output_path,
                hours=self.hours,
                minutes=self.minutes,
                seconds=self.seconds,
                frames=self.frames,
                countdown=self.countdown,
                start_hours=self.start_hours,
                start_minutes=self.start_minutes,
                start_seconds=self.start_seconds,
                start_frames=self.start_frames,
            )

            # Only emit finished if not cancelled
            if not self._cancelled:
                self.finished.emit(self.output_path)
            else:
                # Clean up partial file
                from pathlib import Path
                output_file = Path(self.output_path)
                if output_file.exists():
                    output_file.unlink()
                self.error.emit("Encoding cancelled")

        except Exception as e:
            self.error.emit(str(e))


class EncoderTab(QWidget):
    """
    Encoder tab widget.

    Provides a GUI for all encoder parameters that map to the CLI arguments.
    Uses the existing Encoder class from encoder.py.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[EncoderWorker] = None
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the encoder UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === Start Timecode Section ===
        start_group = QGroupBox("Start Timecode")
        start_layout = QFormLayout()

        self.start_tc_input = QLineEdit()
        self.start_tc_input.setText("1:00:00:00")
        self.start_tc_input.setPlaceholderText("e.g., 1:00:00:00, 15:30:00:00, 1h")
        self.start_tc_input.setToolTip(
            "Starting timecode:\n"
            "• 1:00:00:00 = 1 hour\n"
            "• 15:30:00:00 = 15 hours 30 minutes\n"
            "• 1h = 1 hour (also supports h/m/s/f notation)\n"
            "• In countdown mode, this is disabled and timecode counts from the duration"
        )
        start_layout.addRow("Start:", self.start_tc_input)

        start_group.setLayout(start_layout)
        layout.addWidget(start_group)

        # === Duration Section ===
        duration_group = QGroupBox("Duration")
        duration_layout = QFormLayout()

        self.duration_input = QLineEdit()
        self.duration_input.setText("1:00")
        self.duration_input.setPlaceholderText("e.g., 5m, 1:30, 2h30m, 7h6m5s4f")
        self.duration_input.setToolTip(
            "Duration formats:\n"
            "• 10s = 10 seconds\n"
            "• 5m = 5 minutes\n"
            "• 1h = 1 hour\n"
            "• 1:30 = 1 minute 30 seconds\n"
            "• 1:30:00:15 = 1 min 30 sec 15 frames\n"
            "• 2h30m = 2 hours 30 minutes (compound)\n"
            "• 7h6m5s4f = 7h 6m 5s 4 frames (compound)"
        )
        duration_layout.addRow("Duration:", self.duration_input)

        duration_group.setLayout(duration_layout)
        layout.addWidget(duration_group)

        # === Settings Section ===
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()

        # Frame rate
        self.frame_rate_combo = QComboBox()
        self.frame_rate_combo.addItems([
            "30 fps (Default)",
            "29.97 fps (NTSC NDF)",
            "29.97 fps (NTSC Drop-Frame)",
            "30 fps (Drop-Frame)",
            "25 fps (PAL)",
            "24 fps (Film)",
            "23.98 fps (Film/HD)",
        ])
        self.frame_rate_combo.setCurrentIndex(0)
        self.frame_rate_combo.setMinimumWidth(200)
        settings_layout.addRow("Frame Rate:", self.frame_rate_combo)

        # Sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            "48000 Hz (Default)",
            "44100 Hz",
        ])
        self.sample_rate_combo.setCurrentIndex(0)
        settings_layout.addRow("Sample Rate:", self.sample_rate_combo)

        # Amplitude - use QLineEdit instead of QDoubleSpinBox for better control
        amplitude_layout = QHBoxLayout()
        self.amplitude_input = QLineEdit()
        self.amplitude_input.setText("0.7")
        self.amplitude_input.setPlaceholderText("0.0 - 1.0")
        self.amplitude_input.setMaximumWidth(80)
        self.amplitude_input.setToolTip("Output amplitude (0.0 to 1.0)")

        # Validate input is a number between 0 and 1
        amplitude_validator = QDoubleValidator(0.0, 1.0, 2)
        amplitude_validator.setNotation(QDoubleValidator.StandardNotation)
        self.amplitude_input.setValidator(amplitude_validator)

        amplitude_label = QLabel("0.0 - 1.0")
        amplitude_label.setStyleSheet("color: #888; font-size: 11px;")

        amplitude_layout.addWidget(self.amplitude_input)
        amplitude_layout.addWidget(amplitude_label)
        amplitude_layout.addStretch()

        settings_layout.addRow("Amplitude:", amplitude_layout)

        # Waveform type
        self.waveform_combo = QComboBox()
        self.waveform_combo.addItems([
            "Sine (~25μs rise time, broadcast-friendly)",
            "Square (~1μs rise time)",
        ])
        self.waveform_combo.setCurrentIndex(0)  # Sine is now default
        self.waveform_combo.setMinimumWidth(200)
        self.waveform_combo.setToolTip(
            "Waveform type:\n"
            "• Sine: Smooth transitions, reduces harmonics for broadcast (default)\n"
            "• Square: Fast transitions, traditional for hardware LTC"
        )
        settings_layout.addRow("Waveform:", self.waveform_combo)

        # Countdown mode
        self.countdown_check = QCheckBox("Countdown Mode (timecode counts down to zero)")
        self.countdown_check.setToolTip(
            "When enabled, timecode counts down from the duration.\n"
            "Start timecode is disabled in countdown mode."
        )
        settings_layout.addRow("", self.countdown_check)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === Output Section ===
        output_group = QGroupBox("Output Folder")
        output_layout = QHBoxLayout()

        self.output_path_input = QLineEdit()
        self.output_path_input.setReadOnly(True)
        self.output_path_input.setText("Desktop")

        # Set default to Desktop
        desktop_path = Path.home() / "Desktop"
        self._default_output_dir = str(desktop_path)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_output)

        output_layout.addWidget(self.output_path_input)
        output_layout.addWidget(self.browse_button)
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)

        # === Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # === Status Label ===
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #aaa;")
        layout.addWidget(self.status_label)

        # === Generate Button ===
        self.generate_button = QPushButton("Generate Timecode")
        self.generate_button.setStyleSheet("""
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
        self.generate_button.clicked.connect(self._generate)
        layout.addWidget(self.generate_button)

        # === Stop Button (initially hidden) ===
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #f75c4c;
            }
            QPushButton:pressed {
                background-color: #d73c2c;
            }
        """)
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setVisible(False)
        layout.addWidget(self.stop_button)

        # Add stretch to push everything up
        layout.addStretch()

    def _connect_signals(self):
        """Connect signals for UI interactions."""
        self.countdown_check.toggled.connect(self._on_countdown_toggled)

    def _on_countdown_toggled(self, checked: bool):
        """Handle countdown mode checkbox toggle."""
        if checked:
            # Disable start timecode (but preserve current entry) and darken text
            self.start_tc_input.setEnabled(False)
            self.start_tc_input.setStyleSheet("color: #666;")
            self.start_tc_input.setToolTip(
                "Start timecode is disabled in countdown mode.\n"
                "Timecode counts down from the duration."
            )
        else:
            # Re-enable start timecode and restore text color
            self.start_tc_input.setEnabled(True)
            self.start_tc_input.setStyleSheet("")
            self.start_tc_input.setToolTip(
                "Starting timecode:\n"
                "• 1:00:00:00 = 1 hour\n"
                "• 15:30:00:00 = 15 hours 30 minutes\n"
                "• 1h = 1 hour (also supports h/m/s/f notation)"
            )

    def _browse_output(self):
        """Open folder dialog to select output folder."""
        default_path = self._default_output_dir
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Folder",
            default_path,
        )
        if folder_path:
            self._default_output_dir = folder_path
            # Show folder name in the input
            self.output_path_input.setText(Path(folder_path).name)

    def _get_frame_rate(self) -> tuple[float, bool]:
        """Get frame rate from combo box."""
        text = self.frame_rate_combo.currentText()
        if "29.97" in text and "Drop" in text:
            return 29.97, True
        elif "29.97" in text:
            return 29.97, False
        elif "30 fps" in text and "Drop" in text:
            return 30.0, True
        elif "25" in text:
            return 25.0, False
        elif "24" in text and "23.98" not in text:
            return 24.0, False
        elif "23.98" in text:
            return 23.98, False
        else:
            return 30.0, False

    def _get_sample_rate(self) -> int:
        """Get sample rate from combo box."""
        text = self.sample_rate_combo.currentText()
        if "44100" in text:
            return 44100
        return 48000

    def _get_amplitude(self) -> float:
        """Get amplitude from input field."""
        try:
            return float(self.amplitude_input.text())
        except ValueError:
            return 0.7

    def _get_waveform(self) -> str:
        """Get waveform type from combo box."""
        text = self.waveform_combo.currentText()
        if "Sine" in text:
            return "sine"
        return "square"

    def _generate(self):
        """Start the encoding process."""
        # Validate duration
        duration_str = self.duration_input.text().strip()
        if not duration_str:
            self.status_label.setText("Error: Please enter a duration")
            self.status_label.setStyleSheet("color: #e74c3c;")
            return

        # Parse duration
        try:
            hours, minutes, seconds, frames = parse_timecode(duration_str)
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color: #e74c3c;")
            return

        # Parse start timecode (only if not countdown mode)
        start_str = self.start_tc_input.text().strip()
        start_hours = start_minutes = start_seconds = start_frames = 0
        if start_str and not self.countdown_check.isChecked():
            try:
                start_hours, start_minutes, start_seconds, start_frames = parse_timecode(start_str)
                # Validate start hours (max 639 for extended hours encoding)
                if start_hours > 639:
                    self.status_label.setText(f"Error: Start timecode hours cannot exceed 639 (got {start_hours})")
                    self.status_label.setStyleSheet("color: #e74c3c;")
                    return
            except Exception as e:
                self.status_label.setText(f"Error parsing start timecode: {e}")
                self.status_label.setStyleSheet("color: #e74c3c;")
                return

        # Get frame rate (drop_frame is determined by combo selection)
        frame_rate, drop_frame = self._get_frame_rate()

        # Always auto-generate filename (user only selects output folder)
        # Auto-generate similar to CLI
        if frame_rate == 23.98:
            rate_str = "2398"
        elif frame_rate == 29.97:
            rate_str = "2997"
        else:
            rate_str = str(int(frame_rate))

        drop_suffix = "_drop" if drop_frame else ""
        countdown_prefix = "count_" if self.countdown_check.isChecked() else "ltc_"

        # Format duration string
        if hours > 0:
            duration_str = f"{hours}h"
            if minutes > 0:
                duration_str += f"{minutes}m"
        elif minutes > 0:
            duration_str = f"{minutes}m"
            if seconds > 0:
                duration_str += f"{seconds}s"
        elif seconds > 0:
            duration_str = f"{seconds}s"
        elif frames > 0:
            duration_str = f"{frames}f"
        else:
            duration_str = "0s"

        # Format start timecode if specified (match CLI format: HHMMSSFF)
        start_str = ""
        if start_hours != 0 or start_minutes != 0 or start_seconds != 0 or start_frames != 0:
            start_str = f"_{start_hours:02d}{start_minutes:02d}{start_seconds:02d}{start_frames:02d}"

        output_path = str(Path(self._default_output_dir) / f"{countdown_prefix}{rate_str}fps{drop_suffix}{start_str}_{duration_str}.wav")

        # Disable controls during encoding
        self._set_encoding_state(True)

        # Get other parameters
        sample_rate = self._get_sample_rate()
        amplitude = self._get_amplitude()
        waveform = self._get_waveform()
        countdown = self.countdown_check.isChecked()

        # Start worker thread
        self._worker = EncoderWorker(
            output_path=output_path,
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            frames=frames,
            countdown=countdown,
            start_hours=start_hours,
            start_minutes=start_minutes,
            start_seconds=start_seconds,
            start_frames=start_frames,
            sample_rate=sample_rate,
            frame_rate=frame_rate,
            amplitude=amplitude,
            drop_frame=drop_frame,
            waveform=waveform,
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

        self.status_label.setText(f"Encoding to: {Path(output_path).name}...")
        self.status_label.setStyleSheet("color: #2a82da;")

    def _on_finished(self, output_path: str):
        """Called when encoding finishes successfully."""
        self._set_encoding_state(False)
        self.status_label.setText(f"Generated: {output_path}")
        self.status_label.setStyleSheet("color: #27ae60;")

        # Show message in parent window status bar if available
        parent_window = self.window()
        if hasattr(parent_window, 'show_status'):
            parent_window.show_status(f"Generated: {Path(output_path).name}")

    def _on_error(self, error_msg: str):
        """Called when encoding fails."""
        self._set_encoding_state(False)
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setStyleSheet("color: #e74c3c;")

        # Show error dialog
        parent_window = self.window()
        if hasattr(parent_window, 'show_error'):
            parent_window.show_error("Encoding Error", error_msg)
        else:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Encoding Error", error_msg)

    def _set_encoding_state(self, encoding: bool):
        """Enable/disable controls based on encoding state."""
        self.generate_button.setEnabled(not encoding)
        self.generate_button.setVisible(not encoding)
        self.duration_input.setEnabled(not encoding)
        self.start_tc_input.setEnabled(not encoding and not self.countdown_check.isChecked())
        self.frame_rate_combo.setEnabled(not encoding)
        self.sample_rate_combo.setEnabled(not encoding)
        self.amplitude_input.setEnabled(not encoding)
        self.countdown_check.setEnabled(not encoding)
        self.browse_button.setEnabled(not encoding)
        self.stop_button.setVisible(encoding)
        self.stop_button.setEnabled(encoding)

    def _stop(self):
        """Stop the current encoding operation."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.status_label.setText("Stopping...")
            self.status_label.setStyleSheet("color: #f39c12;")
            self.stop_button.setEnabled(False)
