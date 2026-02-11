# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for FlexTC GUI application.

Build commands:
    Mac:   pyinstaller flextc_gui.spec
    Windows: pyinstaller flextc_gui.spec

Output will be in dist/ directory.
"""

from PySide6.QtCore import qVersion
from pathlib import Path

block_cipher = None

# Collect all imports
a = Analysis(
    ['flextc/gui/__init__.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include assets directory
        ('assets', 'assets'),
    ],
    hiddenimports=[
        # PySide6 modules
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        # Audio libraries
        'numpy',
        'scipy',
        'scipy.signal',
        'sounddevice',
        'soundfile',
        # OSC library
        'python_osc',
        # flextc modules
        'flextc.encoder',
        'flextc.decoder',
        'flextc.biphase',
        'flextc.smpte_packet',
        'flextc.gui.encoder_tab',
        'flextc.gui.decoder_tab',
        'flextc.gui.live_decoder_tab',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FlexTC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Mac-specific
    icon='assets/flextc_icon.png' if Path('assets/flextc_icon.png').exists() else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FlexTC',
)

# Mac app bundle creation
app = BUNDLE(
    exe,
    coll,
    name='FlexTC.app',
    icon='assets/flextc_icon.png',
    bundle_identifier='com.flextc.app',
    info_plist={
        'CFBundleName': 'FlexTC',
        'CFBundleDisplayName': 'FlexTC',
        'CFBundleVersion': '0.1.0',
        'CFBundleShortVersionString': '0.1.0',
        'NSHighResolutionCapable': 'True',
        'LSMinimumSystemVersion': '10.13.0',
        'NSRequiresAquaSystemAppearance': 'False',  # Allow dark mode
        # Microphone permission for live timecode decoding
        'NSMicrophoneUsageDescription': 'FlexTC needs access to the microphone to decode live SMPTE/LTC timecode from audio input.',
    },
)
