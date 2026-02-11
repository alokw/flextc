#!/bin/bash
# Build script for FlexTC GUI on macOS

set -e

echo "Building FlexTC for macOS..."

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip install pyinstaller
fi

# Check if Pillow is installed
if ! python -c "import Pillow" 2>/dev/null; then
    echo "Installing Pillow..."
    pip install pillow
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist

# Build the app
echo "Building app bundle..."
pyinstaller flextc_gui.spec

# Check if build was successful
if [ -d "dist/FlexTC.app" ]; then
    echo "Build successful! App bundle created at: dist/FlexTC.app"

    # Optionally codesign (works on Apple Silicon Macs with ad-hoc signature)
    if command -v codesign &> /dev/null; then
        echo "Code signing with ad-hoc signature..."
        codesign --force --deep --sign - dist/FlexTC.app
        echo "Code signing complete."
    fi

    echo ""
    echo "To run the app:"
    echo "  open dist/FlexTC.app"
    echo ""
    echo "To distribute the app, zip the app bundle:"
    echo "  zip -r FlexTC-macOS.zip dist/FlexTC.app"
else
    echo "Build failed! Check output above for errors."
    exit 1
fi
