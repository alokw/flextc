"""
FlexTC GUI - PySide6-based graphical interface

This module provides a native desktop GUI for FlexTC while keeping
the CLI tools as the primary interface.

GUI architecture:
- gui/__init__.py - Main window and tab container
- gui/encoder_tab.py - Encoder interface (uses existing encoder.Encoder class)
- gui/decoder_tab.py - Decoder interface (uses existing decoder.Decoder class)

The GUI imports and uses the existing Encoder/Decoder classes directly
without any duplication of logic.
"""

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QMessageBox, QStatusBar, QMenuBar, QMenu, QLabel
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QUrl
from PySide6.QtGui import QIcon, QPalette, QColor, QDesktopServices, QPixmap

from flextc.gui.encoder_tab import EncoderTab
from flextc.gui.decoder_tab import DecoderTab
from flextc.gui.live_decoder_tab import LiveDecoderTab


class FlexTCMainWindow(QMainWindow):
    """
    Main application window with tabbed interface for encoder and decoder.

    This is the entry point for the GUI. It creates a tabbed interface
    containing the Encoder and Decoder tabs.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FlexTC - SMPTE Timecode")
        self.resize(700, 650)
        self._version = "2026.02.10"

        # Apply dark theme for pro AV look
        self._apply_dark_theme()

        # Create menu bar
        self._create_menu_bar()

        # Create central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(8, 8, 8, 8)

        # Add logo at the top
        self._add_logo(layout)

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)
        self.tabs.setDocumentMode(True)  # Modern look

        # Create encoder and decoder tabs
        self.encoder_tab = EncoderTab()
        self.decoder_tab = DecoderTab()
        self.live_decoder_tab = LiveDecoderTab()

        # Add tabs
        self.tabs.addTab(self.encoder_tab, "Encoder")
        self.tabs.addTab(self.decoder_tab, "Decode File")
        self.tabs.addTab(self.live_decoder_tab, "Decode Input")

        layout.addWidget(self.tabs)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _apply_dark_theme(self):
        """Apply a professional dark theme suitable for AV tools."""
        palette = QPalette()

        # Dark background colors
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)

        self.setPalette(palette)

        # Additional stylesheet for fine-tuning
        self.setStyleSheet("""
            QMainWindow {
                background-color: #353535;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #353535;
            }
            QTabBar::tab {
                background-color: #444;
                color: #ccc;
                padding: 8px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #555;
                color: #fff;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4a4a4a;
            }
            QGroupBox {
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
                color: #aaa;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #444;
                color: #fff;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #2a82da;
            }
            QPushButton {
                background-color: #444;
                color: #fff;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #555;
            }
            QPushButton:pressed {
                background-color: #2a82da;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #666;
            }
            QCheckBox {
                color: #ccc;
            }
            QLabel {
                color: #ccc;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 3px;
                background-color: #333;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2a82da;
                border-radius: 2px;
            }
        """)

    def _get_asset_path(self, filename: str) -> Path:
        """Get the path to an asset file, handling both development and installed environments."""
        # Try relative to this file first (development)
        asset_path = Path(__file__).parent.parent.parent / "assets" / filename
        if asset_path.exists():
            return asset_path

        # Fallback to package data location
        try:
            from importlib.resources import files
            return files("flextc.assets") / filename
        except Exception:
            # Last resort: return the path anyway and let Qt handle missing file
            return asset_path

    def _add_logo(self, layout: QVBoxLayout):
        """Add the FlexTC logo to the top of the layout."""
        logo_path = self._get_asset_path("flextc_text_dkbg.png")

        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(str(logo_path))
        if not pixmap.isNull():
            # Scale logo to reasonable width while maintaining aspect ratio
            scaled_pixmap = pixmap.scaledToWidth(150, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback to text if image not found
            logo_label.setText("FlexTC")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #ccc;")

        layout.addWidget(logo_label)

    def _create_menu_bar(self):
        """Create the application menu bar with Help and About items."""
        menu_bar = self.menuBar()

        # Help menu
        help_menu = menu_bar.addMenu("&Help")

        # About action
        about_action = help_menu.addAction("&About FlexTC")
        about_action.triggered.connect(self._show_about_dialog)

    def _show_about_dialog(self):
        """Show the About dialog with app info and GitHub link."""
        # Set the icon for the about dialog
        icon_path = self._get_asset_path("flextc_icon.png")

        about_text = f"""
<h3>FlexTC - SMPTE Timecode</h3>
<p><b>Version:</b> {self._version}</p>

<p>A SMPTE/LTC-compatible timecode system that supports both
standard count-up timecode and countdown mode.</p>

<p>Features:</p>
<ul>
<li>Encode and decode SMPTE/LTC timecode</li>
<li>Countdown and count-up modes</li>
<li>Extended hours support (up to 159 hours)</li>
<li>Auto frame rate detection</li>
<li>Auto sample rate detection</li>
<li>OSC re-distribution support</li>
</ul>

<p><a href="https://github.com/alokw/flextc">https://github.com/alokw/flextc</a></p>

<p>License: MIT</p>
"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About FlexTC")
        msg_box.setTextFormat(Qt.RichText)
        msg_box.setText(about_text)
        msg_box.setIcon(QMessageBox.NoIcon)  # Remove default icon so we can use ours

        # Set custom icon (scaled down)
        pixmap = QPixmap(str(icon_path))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            msg_box.setIconPixmap(scaled_pixmap)

        msg_box.exec()

    def show_status(self, message: str, timeout: int = 3000):
        """Show a status message in the status bar."""
        self.status_bar.showMessage(message, timeout)

    def show_error(self, title: str, message: str):
        """Show an error dialog."""
        QMessageBox.critical(self, title, message)

    def show_info(self, title: str, message: str):
        """Show an info dialog."""
        QMessageBox.information(self, title, message)


def main():
    """Main entry point for the GUI application."""
    import sys

    # macOS-specific: Change the menu bar name from "Python" to "FlexTC"
    # This MUST be done before QApplication is created
    if sys.platform == "darwin":
        # Rename the process which affects the menu bar name
        try:
            # Use procname to change the process name visible in the menu bar
            import ctypes
            libSystem = ctypes.CDLL("libSystem.dylib")
            # This is a macOS-specific function to set the process name
            # The prototype is: int setprogname(const char *progname)
            libSystem.setprogname.argtypes = [ctypes.c_char_p]
            libSystem.setprogname(b"FlexTC")
        except Exception:
            # Fallback: just set argv[0]
            sys.argv[0] = "FlexTC"

    app = QApplication(sys.argv)
    app.setApplicationName("FlexTC")
    app.setOrganizationName("FlexTC")
    app.setApplicationDisplayName("FlexTC")
    # On macOS, this changes the menu bar name from "Python" to "FlexTC"
    app.setDesktopFileName("com.flextc.app")

    # Set application icon
    icon_path = Path(__file__).parent.parent.parent / "assets" / "flextc_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    window = FlexTCMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
