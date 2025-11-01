## Test Writing Guidelines (Enhanced)

### Test Structure and Quality
- **Use expected dataframes for assertions**: Instead of multiple `iloc` assertions, create an expected results dataframe and use `assert_frame_equal`
  ```python
  # ❌ Avoid this:
  assert result.iloc[0]["column"] == expected_value
  assert result.iloc[1]["column"] == other_value
  
  # ✅ Prefer this:
  expected_df = pd.DataFrame({
      "column": [expected_value, other_value]
  })
  assert_frame_equal(result, expected_df)
  ```

- **Only write meaningful tests**: Before writing a test, ask "What is this actually testing?"
  - ❌ Don't test library behavior (e.g., matplotlib's error handling with empty dataframes)
  - ❌ Don't write redundant tests that verify the same thing multiple times
  - ✅ Test your code's logic, edge cases, and data transformations

- **Avoid test overkill**: If a test feels redundant with existing tests, remove it
  - Example: If you're already testing kwargs are passed correctly, don't write additional tests for each specific kwarg

- **Make test assertions specific**: Tests should verify actual behavior, not just object existence
  ```python
  # ❌ Weak assertion:
  assert box is not None
  
  # ✅ Strong assertion:
  bbox = box.get_path().get_extents()
  assert bbox.width == pytest.approx(3.5, abs=0.01)
  assert bbox.height == pytest.approx(2.5, abs=0.01)
  ```

- **Extract magic numbers as constants**: If tests or code use threshold values, define them as class/module constants
  ```python
  # In the class:
  GREEN_THRESHOLD = 1.0
  RED_THRESHOLD = -1.0
  
  # Then use them:
  if value > self.GREEN_THRESHOLD:
      color = "green"
  ```

### Test Docstrings
- Make test docstrings descriptive and specific
- ❌ Avoid vague terms: "Tests edge cases"
- ✅ Be specific: "Tests behavior when input dataframe has duplicate index values"

## PR Scope and Cleanup Guidelines

### What to Exclude from PRs
- **Documentation assets**: Don't include example images, plots, or extensive documentation updates
  - These belong in dedicated documentation PRs or a plots gallery
  - Exception: The single plot/image needed for the docs page is acceptable

- **Unnecessary imports**: Remove all unused imports before submitting
  - Check `pyproject.toml`, test files, and source files

- **Out-of-scope changes**: Keep PRs focused on the feature/fix described in the title
  - If you make improvements to unrelated code, consider a separate PR

### Before Submitting a PR
1. Review all files in the PR and ask: "Does this file need to be in this PR?"
2. Remove any files that don't directly support the PR's purpose
3. Clean up any experimental or debugging code

## Code Quality Guidelines

### Prefer Vectorized Operations
- Use pandas/numpy vectorized operations instead of loops or list comprehensions
  ```python
  # ❌ Avoid:
  x_labels = [str(col) for col in df.columns]
  y_labels = [str(idx) for idx in df.index]
  
  # ✅ Prefer:
  x_labels = df.columns.astype(str).to_list()
  y_labels = df.index.astype(str).to_list()
  ```

- When working with arrays, use numpy operations:
  ```python
  # ❌ Avoid:
  for i, point in enumerate(points):
      x_coords.append(point[0])
      y_coords.append(point[1])
  
  # ✅ Prefer:
  x_coords = data_df[x_col].to_numpy()
  y_coords = data_df[y_col].to_numpy()
  ```

### Code Simplification
- Look for opportunities to simplify complex expressions
- Combine related operations into single, clear statements
- Don't initialize variables unnecessarily (let assignment handle it)
  ```python
  # ❌ Avoid:
  x_coords = []
  y_coords = []
  x_coords = data_df[x_col].to_numpy()
  y_coords = data_df[y_col].to_numpy()
  
  # ✅ Prefer:
  x_coords = data_df[x_col].to_numpy()
  y_coords = data_df[y_col].to_numpy()
  ```

## API Design Guidelines

### Parameter Flexibility
- When you hardcode a value that might vary by use case, consider making it an optional parameter
- Example: Format strings, thresholds, or display options should often be parameters
  ```python
  # ❌ Hardcoded:
  def plot_heatmap(data):
      format_str = ".1f"  # Always uses this format
  
  # ✅ Flexible:
  def plot_heatmap(data, value_format: str = ".1f"):
      # Now users can customize if needed
  ```

- When changing existing parameters, be careful with defaults
  - If a parameter should default to `True`, keep it as `True`
  - Document why you're changing a default if needed

### Validation and Error Handling
- Validate inputs early, especially when nodes/references might not exist
- Provide clear error messages with available options when validation fails
  ```python
  if child_id not in node_centers:
      available_nodes = list(node_centers.keys())
      raise ValueError(
          f"Child node '{child_id}' not found. Available nodes: {available_nodes}"
      )
  ```
