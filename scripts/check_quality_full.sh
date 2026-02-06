#!/bin/bash
# Extended quality checks with type checking and deep analysis
# More thorough but slower than check_quality.sh

set -e

echo "🔍 VisionForge Extended Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📝 Black (code formatting)..."
black --check --diff orchard/ tests/ forge.py
echo "✓ Black passed"
echo ""

echo "📦 isort (import sorting)..."
isort --check-only --diff orchard/ tests/ forge.py
echo "✓ isort passed"
echo ""

echo "✨ Flake8 (linting)..."
flake8 orchard/ tests/ forge.py --max-line-length=100 --extend-ignore=E203,W503
echo "✓ Flake8 passed"
echo ""

echo "🔒 Bandit (security linting)..."
bandit -r orchard/ -ll -q
echo "✓ Bandit passed"
echo ""

echo "🔍 MyPy (type checking)..."
echo "  Installing type stubs..."
pip install -q types-PyYAML types-requests 2>/dev/null || true
mypy orchard/ --ignore-missing-imports --no-strict-optional
echo "✓ MyPy passed"
echo ""

echo "📊 Radon (complexity analysis)..."
echo "  Cyclomatic Complexity (max: B):"
radon cc orchard/ -n B --total-average
echo ""
echo "  Maintainability Index (min: B):"
radon mi orchard/ -n B
echo ""
echo "  Raw Metrics:"
radon raw orchard/ -s
echo "✓ Radon passed"
echo ""

echo "🧪 Pytest (tests + coverage)..."
pytest --cov=orchard --cov-report=term-missing --cov-report=html -v tests/
echo ""

echo "✅ All extended quality checks passed!"
echo ""
echo "📊 Coverage report: htmlcov/index.html"
