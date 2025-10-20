#!/usr/bin/env python3
"""
Setup script for Futures Roll Analysis Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="futures-roll-analysis",
    version="1.0.0",
    author="Austin Li",
    description="Framework for analyzing institutional roll patterns in futures markets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/futures-roll-analysis",  # Update if you have a repo
    
    packages=find_packages(where="etf_roll_analysis"),
    package_dir={"": "etf_roll_analysis"},
    
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "pyyaml>=5.4.0",
        "python-dateutil>=2.8.0",
    ],
    
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "ipython>=7.20.0",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
        ],
    },
    
    python_requires=">=3.8",
    
    entry_points={
        "console_scripts": [
            "organize-futures=organize_data:main",
            "roll-analyze=etf_roll_analysis.scripts.analyze:main",
            # Backward-compat shims (optional):
            "analyze-hourly=etf_roll_analysis.scripts.analyze:main",
            "hg-analysis=etf_roll_analysis.scripts.analyze:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",  # Update as appropriate
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    keywords="futures trading roll analysis commodities finance quantitative",
    
    project_urls={
        "Documentation": "https://github.com/yourusername/futures-roll-analysis",
        "Source": "https://github.com/yourusername/futures-roll-analysis",
    },
)
