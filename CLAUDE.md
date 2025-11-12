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
- Using double quotes when quoting strings
- When checking for None values use `if my_var is not None` rather than `if my_var:`
- When checking for empty lists values use `if len(my_list) > 0` rather than `if my_list:`
- Always create "conventional commit" commit messages
- Do not stage files or commit unless asked to
- Docstrings should specify argument and return types
- Type annotations should use Python 3.11 formats
- Remove unnecessary trailing whitespace
- Use vectorized operations for numpy arrays and pandas DataFrames/Series wherever possible instead of loops or
  .apply() to keep code clean and performant. Avoid converting Series to lists for iteration or using .iterrows() -
  use vectorized operations directly on the DataFrame/Series
- Extract magic numbers as named constants with descriptive names (e.g., `GREEN_THRESHOLD = 1.0` instead of hardcoded
  `1.0` in conditionals)
- Add input validation for public functions and class methods: validate types, ranges, and constraints early with clear
  error messages
- Use ternary operators for simple conditional assignments (e.g., `status = "active" if count > 0 else "inactive"`
  instead of multi-line if/else blocks)

## Test Writing Guidelines

### Core Principles

- **Test package functionality, not libraries**: Every test must import and exercise PyRetailScience code. Do not test
  pandas, Python built-ins, or third-party libraries in isolation.
- **Verify behavior, not execution**: Tests must have meaningful assertions that validate the actual behavior claimed by
  the test name. Checking return types or "not None" is insufficient.
- **Test outputs, not implementation**: Focus on public API behavior and results. Avoid testing internal state, private
  attributes, or "how" code works internally. Testing private method outputs/behavior is acceptable (e.g.,
  `obj._calculate(5) == 10`), but testing internal state is not (e.g., `obj._cache == {...}`).
- **Use realistic retail data**: Test data should reflect real retail scenarios (customer_id, store_id, transaction
  amounts) rather than generic placeholders ("A", "B", "test", "data").

### Required Practices

- Prefer test classes for organizing related tests (group tests by module or class being tested)
- Always put imports at the top of the module (never import inside test methods)
- Where possible prefer pandas `assert_frame_equal` to asserting individual values of the dataframe
    - This may require creating an expected dataframe to compare against
- Use `pytest.mark.parametrize` when testing multiple input variations of the same behavior
- Test files should follow pattern: `test_*.py` with test functions `test_*`
- Test names must accurately describe what is being tested
- Each test should have at least one meaningful assertion that validates the claimed behavior
- Include boundary/edge case tests for threshold values, limits, and special cases
- When testing against expected values (colors, formats, etc.), reference package constants rather than hardcoding
  values in tests
- Use pytest fixtures for shared test data setup to improve readability and reduce duplication

### Anti-Patterns to Avoid

- **Don't test without package imports**: If no PyRetailScience code is imported, the test is invalid
- **Don't test Python/library basics**: Avoid tests that verify fundamental Python or standard library behavior (set
  operations, dict.get(), list methods, string operations) without exercising PyRetailScience functionality. These tests
  belong in Python's own test suite.
- **Don't write assertion-free tests**: Every test must have assertions that can actually fail. Avoid:
    - Tests with no assertions at all
    - Only `assert obj is not None` after object creation (always true)
    - Only `assert hasattr()` or `callable()` checks without behavior verification
    - Only `assert isinstance()` type checks without validating actual behavior
    - Assertions that can never fail (e.g., `assert x >= 0 or x < 0`)
    - Vague comparison assertions that only check "something changed" (e.g., `assert filtered_df.shape !=
      unfiltered_df.shape`). Better: verify the actual filtering behavior (e.g., `assert len(filtered_df) == 5` and
      `assert (filtered_df['price'] > 100).all()`)
- **Don't duplicate tests**: Multiple tests with identical structure should use parametrize instead
- **Don't over-mock**: Mock external dependencies only (databases, APIs, file I/O, network calls). Never mock internal
  PyRetailScience logic or calculations. Test real package behavior.
- **Don't write trivial tests**: Avoid testing constants equal their definitions, class names, or tautological
  assertions (True == True)
- **Don't test implementation details**: Avoid asserting internal state values or data structures. Testing private
  method behavior/outputs is OK; testing how data is stored internally is not.
- **Don't use generic test data**: Use domain-specific examples (retail customer data, not "A", "B", "C")
- **Don't mismatch test names and assertions**: If test claims to verify sorting, actually assert sort order. If it
  claims validation, verify validation logic works.

### Before Committing Tests

Before committing test code, verify each test meets these criteria:

1. **Imports PyRetailScience code**: At least one import from `pyretailscience.*`
2. **Calls package functionality**: Test actually invokes package functions/classes/methods
3. **Has meaningful assertions**: Assertions verify actual behavior, not just types or existence
4. **Test name matches behavior**: Assertions validate what the test name claims to test
5. **No substantial duplication**: Similar tests use parametrize or are combined
6. **Uses realistic data**: Test data reflects retail domain (customer IDs, prices, stores, etc.)
7. **Minimal mocking**: Only external dependencies are mocked, not internal package logic
