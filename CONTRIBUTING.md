# Contributing to Distribird

Contributions are welcome! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/HUN-REN-AI1Science/Distribird.git
cd distribird
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

We use ruff for linting and formatting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

Type checking with mypy:
```bash
mypy src/distribird/
```

## Pull Requests

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Reporting Issues

Please open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
