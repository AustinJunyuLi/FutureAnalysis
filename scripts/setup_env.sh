#!/bin/bash
# Environment setup script for futures roll analysis

echo "🔧 Setting up Python environment for futures roll analysis..."

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Conda paths
CONDA_BASE="/home/austinli/miniconda3"
CONDA_ENV="futures-roll"
CONDA_PYTHON="$CONDA_BASE/envs/$CONDA_ENV/bin/python"

# Check if conda environment exists
if [ ! -f "$CONDA_PYTHON" ]; then
    echo "❌ Error: Conda environment 'futures-roll' not found!"
    echo "   Please create it first with: conda create -n futures-roll python=3.11"
    return 1 2>/dev/null || exit 1
fi

# Export paths
export PATH="$CONDA_BASE/envs/$CONDA_ENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Set aliases to ensure correct Python is used
alias python="$CONDA_PYTHON"
alias pip="$CONDA_PYTHON -m pip"
alias pytest="$CONDA_PYTHON -m pytest"

# Verification
echo ""
echo "✅ Environment configured:"
echo "  Python: $(which python)"
echo "  Version: $(python --version)"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""

# Check critical packages
echo "📦 Checking packages..."
python -c "
import sys
packages = ['pandas', 'numpy', 'pyarrow', 'yaml', 'pytest']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✓ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'  ✗ {pkg} (missing)')

if missing:
    print(f'\n⚠️  Missing packages: {', '.join(missing)}')
    print(f'   Install with: pip install {' '.join(missing)}')
else:
    print('\n✅ All required packages installed!')
" 2>/dev/null

echo ""
echo "🚀 Ready to use! Try these commands:"
echo "  python -m futures_roll_analysis.cli.hourly --help"
echo "  python -m pytest tests/"
echo "  python -c 'from futures_roll_analysis import __version__; print(__version__)'"
