#!/bin/bash
# Quality checks automation script
# Run all code quality checks in one go

set -e

echo "🔍 VisionForge Quality Checks"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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

echo "📊 Radon (complexity analysis)..."
echo "  Cyclomatic Complexity (max: B):"
radon cc orchard/ -n B --total-average
echo ""
echo "  Maintainability Index (min: B):"
radon mi orchard/ -n B
echo "✓ Radon passed"
echo ""

echo "🧪 Pytest (tests + coverage)..."
pytest --cov=orchard --cov-report=term-missing -v tests/
echo ""

echo "✅ All quality checks passed!"
