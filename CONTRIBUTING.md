# Contributing to VisionForge

Thank you for considering contributing to VisionForge!

## Project Direction

Project goals and roadmap are defined in the [main README](README.md). Before proposing new features, please review the stated objectives to ensure alignment with the project's vision.

**Contributions are welcome in these areas:**
- Bug fixes and behavior improvements
- Performance optimizations
- Test coverage enhancements
- Documentation improvements
- Code quality improvements

**For new features:** Please open an issue first to discuss alignment with project goals.

## Code Quality Standards

All contributions must maintain the project's quality standards:

### 1. Testing Requirements

- **All code changes must include tests**
- **Maintain 100% test coverage** (current standard)
- Tests must pass locally before submitting

```bash
# Run test suite
pytest tests/ -v

# Check coverage
pytest tests/ --cov=orchard --cov-report=term-missing
```

### 2. Quality Checks

Before committing, run the automated quality checks:

```bash
# Run all checks (linting, formatting, tests, coverage)
bash scripts/run_checks.sh
```

This script validates:
- Code formatting (Black)
- Linting (Flake8)
- Type hints (MyPy)
- Test coverage (100% required)

### 3. Code Style

- Follow existing code patterns and architecture
- Use type hints for all function signatures
- Write clear docstrings (Google style)
- Keep functions focused and testable

## Contribution Workflow

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b fix/issue-description`)
3. **Make your changes** with tests
4. **Run quality checks** (`bash scripts/run_checks.sh`)
5. **Commit** with clear messages
6. **Push** and open a Pull Request

## Questions?

Open an issue for:
- Feature proposals (discuss before implementing)
- Bug reports (include reproduction steps)
- Architecture questions
- Documentation clarifications

---

**Note:** This is a personal research project. Contributions are appreciated, but the maintainer reserves final decisions on features and direction.
