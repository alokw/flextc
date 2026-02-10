"""
Decoder Tab - GUI interface for timecode decoding

This tab provides a graphical interface for the decoder functionality.
It imports and uses the existing decode_file function from decoder.py.
"""

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLineEdit, QComboBox, QSpinBox, QPushButton, QFileDialog,
    QProgressBar, QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView
)
from PySide6.QtCore import Qt, QThread, Signal

from flextc.decoder import decode_file, format_timecode
from flextc.smpte_packet import Timecode


class DecoderWorker(QThread):
    """
    Worker thread for decoding timecode files.

    Runs the decoding operation in a separate thread to keep
    the GUI responsive during long operations.
    """

    progress = Signal(str)  # Status message
    finished = Signal(dict)  # Result dict on success
    error = Signal(str)     # Error message on failure

    def __init__(self, file_path: str, sample_rate: int, frame_rate: Optional[float]):
        super().__init__()
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate

    def run(self):
        """Run the decoding operation in the background thread."""
        try:
            self.progress.emit("Reading file...")

            # Use existing decode_file function
            first_tc, last_tc, detected_rate, duration = decode_file(
                self.file_path,
                self.sample_rate,
                self.frame_rate
            )

            if first_tc is None:
                self.error.emit("No SMPTE/LTC frames detected in file.")
                return

            # Determine direction by comparing timecodes
            fps = first_tc.fps

            def tc_to_frames(tc):
                total = (tc.hours * 3600 + tc.minutes * 60 + tc.seconds) * fps + tc.frames
                return int(total)

            first_frames = tc_to_frames(first_tc)
            if last_tc is not None:
                last_frames = tc_to_frames(last_tc)
                counting_up = last_frames >= first_frames
            else:
                counting_up = first_tc.count_up
                last_frames = first_frames

            # Calculate duration
            if counting_up:
                duration_frames = last_frames - first_frames
            else:
                duration_frames = first_frames - last_frames

            # Convert duration to HH:MM:SS:FF
            dur_hours = int(duration_frames // (fps * 3600))
            duration_frames %= fps * 3600
            dur_minutes = int(duration_frames // (fps * 60))
            duration_frames %= fps * 60
            dur_seconds = int(duration_frames // fps)
            dur_ff = int(duration_frames % fps)

            separator = ";" if first_tc.is_drop_frame else ":"
            tc_duration = f"{dur_hours:02d}:{dur_minutes:02d}:{dur_seconds:02d}{separator}{dur_ff:02d}"

            # Build result dictionary
            result = {
                'first_tc': first_tc,
                'last_tc': last_tc,
                'frame_rate': detected_rate,
                'duration_seconds': duration,
                'counting_up': counting_up,
                'tc_duration': tc_duration,
                'file_size': Path(self.file_path).stat().st_size,
            }

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class DecoderTab(QWidget):
    """
    Decoder tab widget.

    Provides a GUI for file-based decoding that uses the existing
    decode_file function from decoder.py.

    Note: Live decoding is not implemented in the GUI yet,
    as it requires continuous audio input handling which is
    better suited to the CLI interface.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._worker: Optional[DecoderWorker] = None
        self._setup_ui()

    def _setup_ui(self):
        """Set up the decoder UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # === File Selection Section ===
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()

        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("Select a WAV file to decode...")
        self.file_input.setReadOnly(True)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_file)

        file_layout.addWidget(self.file_input)
        file_layout.addWidget(self.browse_button)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # === Settings Section ===
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()

        # Sample rate
        self.sample_rate_combo = QComboBox()
        self.sample_rate_combo.addItems([
            "48000 Hz (Default)",
            "44100 Hz",
        ])
        self.sample_rate_combo.setCurrentIndex(0)
        settings_layout.addRow("Sample Rate:", self.sample_rate_combo)

        # Frame rate (optional, for auto-detect leave as is)
        self.frame_rate_combo = QComboBox()
        self.frame_rate_combo.addItems([
            "Auto-detect (Recommended)",
            "23.98 fps",
            "24 fps",
            "25 fps",
            "29.97 fps",
            "30 fps",
        ])
        self.frame_rate_combo.setCurrentIndex(0)
        settings_layout.addRow("Frame Rate:", self.frame_rate_combo)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # === Decode Button ===
        self.decode_button = QPushButton("Decode File")
        self.decode_button.setStyleSheet("""
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
        self.decode_button.clicked.connect(self._decode)
        layout.addWidget(self.decode_button)

        # === Progress Bar ===
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        # === Results Section ===
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        # Set minimum height to show all rows without scrolling
        self.results_table.setMinimumHeight(250)
        self.results_table.setStyleSheet("""
            QTableWidget {
                background-color: #333;
                gridline-color: #444;
            }
            QTableWidget::item {
                padding: 4px;
            }
            QTableWidget::item:alternate {
                background-color: #3a3a3a;
            }
        """)

        # Initialize with empty rows
        self._init_results_table()

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Add stretch to push everything up
        layout.addStretch()

    def _init_results_table(self):
        """Initialize the results table with empty values."""
        labels = [
            "File Name",
            "File Size",
            "Frame Rate",
            "Start Timecode",
            "End Timecode",
            "Duration",
            "Direction",
            "Drop-Frame",
        ]

        self.results_table.setRowCount(len(labels))
        for i, label in enumerate(labels):
            self.results_table.setItem(i, 0, QTableWidgetItem(label))
            self.results_table.setItem(i, 1, QTableWidgetItem("—"))

    def _browse_file(self):
        """Open file dialog to select input file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Timecode File",
            "",
            "WAV Files (*.wav);;All Files (*)"
        )
        if file_path:
            self.file_input.setText(file_path)
            # Clear previous results
            self._init_results_table()

    def _get_sample_rate(self) -> int:
        """Get sample rate from combo box."""
        text = self.sample_rate_combo.currentText()
        if "44100" in text:
            return 44100
        return 48000

    def _get_frame_rate(self) -> Optional[float]:
        """Get frame rate from combo box, or None for auto-detect."""
        text = self.frame_rate_combo.currentText()
        if "Auto" in text:
            return None
        elif "23.98" in text:
            return 23.98
        elif "24 fps" in text:
            return 24.0
        elif "25" in text:
            return 25.0
        elif "29.97" in text:
            return 29.97
        elif "30" in text:
            return 30.0
        return None

    def _decode(self):
        """Start the decoding process."""
        # Validate file input
        file_path = self.file_input.text().strip()
        if not file_path:
            self._show_result_row(0, "Error", "Please select a file")
            return

        if not Path(file_path).exists():
            self._show_result_row(0, "Error", f"File not found: {Path(file_path).name}")
            return

        # Get parameters
        sample_rate = self._get_sample_rate()
        frame_rate = self._get_frame_rate()

        # Disable controls during decoding
        self._set_decoding_state(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress

        # Start worker thread
        self._worker = DecoderWorker(file_path, sample_rate, frame_rate)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, message: str):
        """Handle progress updates."""
        # Could update a status label here
        pass

    def _on_finished(self, result: dict):
        """Called when decoding finishes successfully."""
        self._set_decoding_state(False)
        self.progress_bar.setVisible(False)

        first_tc: Timecode = result['first_tc']
        last_tc: Timecode = result.get('last_tc')
        file_path = self.file_input.text()

        # Update results table
        self._show_result_row(0, "File Name", Path(file_path).name)

        # File size
        size_bytes = result['file_size']
        if size_bytes >= 1024 * 1024:
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        elif size_bytes >= 1024:
            size_str = f"{size_bytes / 1024:.2f} KB"
        else:
            size_str = f"{size_bytes} bytes"
        self._show_result_row(1, "File Size", size_str)

        # Frame rate
        self._show_result_row(2, "Frame Rate", f"{result['frame_rate']} fps")

        # Start timecode
        self._show_result_row(3, "Start Timecode", format_timecode(first_tc))

        # End timecode
        if last_tc:
            separator = ";" if first_tc.is_drop_frame else ":"
            direction_symbol = "▲" if result['counting_up'] else "▼"
            end_tc_str = f"{direction_symbol} {last_tc.hours:02d}:{last_tc.minutes:02d}:{last_tc.seconds:02d}{separator}{last_tc.frames:02d}"
            self._show_result_row(4, "End Timecode", end_tc_str)
        else:
            self._show_result_row(4, "End Timecode", "(not found)")

        # Duration
        self._show_result_row(5, "Duration", result['tc_duration'])

        # Direction
        direction = "Counting up" if result['counting_up'] else "Counting down"
        self._show_result_row(6, "Direction", direction)

        # Drop-frame
        self._show_result_row(7, "Drop-Frame", "Yes" if first_tc.is_drop_frame else "No")

        # Show success in parent window status bar
        parent_window = self.window()
        if hasattr(parent_window, 'show_status'):
            parent_window.show_status(f"Decoded: {Path(file_path).name}")

    def _on_error(self, error_msg: str):
        """Called when decoding fails."""
        self._set_decoding_state(False)
        self.progress_bar.setVisible(False)
        self._show_result_row(0, "Error", error_msg)

        # Show error dialog
        parent_window = self.window()
        if hasattr(parent_window, 'show_error'):
            parent_window.show_error("Decoding Error", error_msg)

    def _show_result_row(self, row: int, label: str, value: str):
        """Update a row in the results table."""
        self.results_table.setItem(row, 0, QTableWidgetItem(label))
        self.results_table.setItem(row, 1, QTableWidgetItem(value))

    def _set_decoding_state(self, decoding: bool):
        """Enable/disable controls based on decoding state."""
        self.decode_button.setEnabled(not decoding)
        self.browse_button.setEnabled(not decoding)
        self.sample_rate_combo.setEnabled(not decoding)
        self.frame_rate_combo.setEnabled(not decoding)
