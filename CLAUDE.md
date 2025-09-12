# PyRetailScience Development Guide

## Build & Test Commands

- Install dependencies: `uv sync`
- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest tests/test_file.py::test_function_name`
- Run tests with pattern: `uv run pytest -k "pattern"`
- Run tests with coverage: `uv run pytest --cov=pyretailscience`
- Lint code: `uv run ruff check .`
- Format code: `uv run ruff format .`
- Run notebook: `uv run jupyter nbconvert --to notebook --execute my_notebook.ipynb`

## Code Style Guidelines

- Follow Google Python Style Guide
- Use Google-style docstrings with Args, Returns, and Raises sections
- Import order: standard library → third-party → internal packages
- Use explicit type annotations for all functions and parameters
- Class names use CamelCase; functions, methods, variables use snake_case
- Private methods/functions should be prefixed with underscore (_)
- Max line length: 120 characters
- Error handling: validate inputs early, use specific exceptions with descriptive messages
- File naming: Module names should be lowercase with underscores
- Test files should follow pattern: `test_*.py` with test functions `test_*`
- Using double quotes when quoting strings
- When checking for None values use `if my_var is not None` rather than `if my_var:`
- When checking for empty lists values use `if len(my_list) > 0` rather than `if my_list:`
- Always create "conventional commit" commit messages
- Prefer test classes for modules or classes
- Do not stage files or commit unless asked to
- Docstrings should specify argument and return types
- Type annotations should use Python 3.11 formats
- Remove unnecessary trailing whitespace
