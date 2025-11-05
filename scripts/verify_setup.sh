#!/bin/bash
# Verification script for futures roll analysis environment setup

echo "🔍 Futures Roll Analysis - Environment Verification"
echo "===================================================="
echo ""

# Check Python environment
echo "📍 Python Environment:"
echo "  Python: $(/home/austinli/miniconda3/envs/futures-roll/bin/python --version)"
echo "  Location: /home/austinli/miniconda3/envs/futures-roll/bin/python"
echo ""

# Check packages
echo "📦 Core Packages:"
/home/austinli/miniconda3/envs/futures-roll/bin/python -c "
import pandas
import numpy
import pyarrow
import pytest
print(f'  ✓ pandas {pandas.__version__}')
print(f'  ✓ numpy {numpy.__version__}')
print(f'  ✓ pyarrow {pyarrow.__version__}')
print(f'  ✓ pytest {pytest.__version__}')
" 2>/dev/null || echo "  ✗ Some packages missing"
echo ""

# Check CLI commands
echo "🔧 CLI Commands:"
if [ -f /home/austinli/miniconda3/envs/futures-roll/bin/futures-roll ]; then
    echo "  ✓ futures-roll (unified CLI)"
else
    echo "  ✗ futures-roll not found"
fi

if [ -f /home/austinli/miniconda3/envs/futures-roll/bin/futures-roll-hourly ]; then
    echo "  ✓ futures-roll-hourly (legacy)"
else
    echo "  ✗ futures-roll-hourly not found"
fi
echo ""

# Check project structure
echo "📁 Project Structure:"
cd /home/austinli/Dropbox/futures_individual_contracts_1min 2>/dev/null && {
    echo "  ✓ Project directory accessible"
    echo "  ✓ Source files: $(find src -name "*.py" -type f | wc -l) Python files"
    echo "  ✓ Test files: $(find tests -name "*.py" -type f | wc -l) test files"
    echo "  ✓ Data files: $(find organized_data -name "*.txt" -type f 2>/dev/null | wc -l) raw files"
} || echo "  ✗ Project directory not found"
echo ""

# Test imports
echo "🧪 Testing Package Import:"
/home/austinli/miniconda3/envs/futures-roll/bin/python -c "
from futures_roll_analysis import __version__
print(f'  ✓ Package version: {__version__}')
" 2>/dev/null || echo "  ✗ Import failed"
echo ""

# Show new CLI usage
echo "📘 New Unified CLI Usage:"
echo "  futures-roll analyze --mode hourly   # Run hourly analysis"
echo "  futures-roll analyze --mode daily    # Run daily analysis"
echo "  futures-roll organize                # Organize raw files"
echo "  futures-roll --help                  # Show all commands"
echo ""

echo "✅ Verification complete!"
