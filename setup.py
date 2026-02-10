"""
FlexTC - SMPTE-Compatible Bidirectional Timecode
Setup configuration.
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="flextc",
    version="0.1.0",
    description="SMPTE/LTC-compatible bidirectional timecode system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="FlexTC Contributors",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "flextc-encode=flextc.encoder:main",
            "flextc-decode=flextc.decoder:main",
            "flextc-gui=flextc.gui:main",
        ],
    },
    extras_require={
        "gui": ["PySide6>=6.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: System :: Networking",
    ],
)
