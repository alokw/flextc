@echo off
REM Build script for FlexTC GUI on Windows

echo Building FlexTC for Windows...

REM Check if PyInstaller is installed
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    pip install pyinstaller
)

REM Clean previous builds
echo Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

REM Build the app
echo Building Windows executable...
pyinstaller flextc_gui.spec

REM Check if build was successful
if exist "dist\FlexTC\FlexTC.exe" (
    echo Build successful! Executable created at: dist\FlexTC\FlexTC.exe
    echo.
    echo To run the app:
    echo   dist\FlexTC\FlexTC.exe
    echo.
    echo To distribute the app, create a ZIP of the dist\FlexTC folder
) else (
    echo Build failed! Check output above for errors.
    exit /b 1
)
