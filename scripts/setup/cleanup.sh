#!/bin/bash
# Cleanup script for futures roll analysis project

echo "🧹 Starting cleanup of futures roll analysis project..."

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# 1. Clean Python cache files
echo "📦 Removing Python cache files..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
find . -type f -name ".coverage" -delete 2>/dev/null
find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null
echo "  ✓ Cache files removed"

# 2. Clean test outputs
echo "🧪 Cleaning test outputs..."
rm -rf outputs/test_run* 2>/dev/null
rm -rf outputs/debug_run* 2>/dev/null
rm -rf outputs/vscode_run* 2>/dev/null
rm -rf outputs/test_validation* 2>/dev/null
echo "  ✓ Test outputs cleaned"

# 3. Report on file counts
echo ""
echo "📊 Project Statistics:"
echo "  Python source files: $(find src -name "*.py" -type f | wc -l)"
echo "  Test files: $(find tests -name "*.py" -type f | wc -l)"
echo "  Total LOC: $(find src -name "*.py" -type f -exec cat {} + | wc -l)"
echo "  Data files: $(find organized_data -name "*.txt" -type f 2>/dev/null | wc -l)"

# 4. Check for large files
echo ""
echo "📁 Large files (>1MB):"
find . -type f -size +1M ! -path "./organized_data/*" ! -path "./outputs/*" ! -path "./.git/*" -exec du -h {} + 2>/dev/null | sort -rh | head -10

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "💡 Next steps:"
echo "  1. Review and merge CODE_REVIEW.md into README.md"
echo "  2. Run: source scripts/setup_env.sh"
echo "  3. Test with: python -m pytest tests/"
