# Technical Specification: Regression Line Enhancement

**Feature**: Expand `add_regression_line()` function in `pyretailscience/plots/styles/graph_utils.py`
**Branch**: `feature/enhance-regression-line-capabilities`
**Date**: 2025-01-18
**Version**: 1.0

## 📋 Overview

Enhance the existing linear regression functionality to support multiple regression algorithms while maintaining full backward compatibility and following established pyretailscience patterns.

## 🎯 Design Principles

1. **Backward Compatibility**: All existing calls continue to work unchanged
2. **Consistent API**: Follow existing pyretailscience function patterns
3. **Mathematical Accuracy**: Use established scipy/sklearn algorithms
4. **Retail Focus**: Prioritize algorithms useful for retail analytics
5. **Error Resilience**: Graceful handling of edge cases and data issues

## 🔧 API Specification

### Enhanced Function Signature

```python
def add_regression_line(
    ax: Axes,
    regression_type: Literal["linear", "power", "logarithmic", "exponential", "polynomial"] = "linear",
    color: str = "red",
    linestyle: str = "--",
    text_position: float = 0.6,
    show_equation: bool = True,
    show_r2: bool = True,
    # New parameters for advanced regression types
    polynomial_degree: int = 2,
    confidence_interval: float | None = None,
    **kwargs: dict[str, any],
) -> Axes:
```

### Parameter Specifications

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `regression_type` | `Literal[...]` | `"linear"` | Algorithm selection (NEW) |
| `polynomial_degree` | `int` | `2` | Degree for polynomial regression (NEW) |
| `confidence_interval` | `float \| None` | `None` | Show confidence bands (0.0-1.0) (NEW) |
| All existing params | - | Same | Unchanged for compatibility |

## 📊 Regression Type Specifications

### 1. Linear Regression (Default)
```python
# Usage: regression_type="linear" (default)
# Algorithm: scipy.stats.linregress()
# Equation: y = mx + b
# Use Case: Basic trend analysis
```

**Mathematical Implementation:**
- Method: Ordinary Least Squares (OLS)
- Library: `scipy.stats.linregress()`
- Equation Format: `y = {slope:.3f}x + {intercept:.3f}`
- R² Calculation: `r_value**2`

### 2. Power Law Regression
```python
# Usage: regression_type="power"
# Algorithm: Log-log transformation + linear regression
# Equation: y = ax^b → log(y) = log(a) + b*log(x)
# Use Case: Price elasticity, scaling relationships
```

**Mathematical Implementation:**
```python
# Transform to log space
log_x = np.log(x_data[x_data > 0])
log_y = np.log(y_data[y_data > 0])

# Linear regression in log space
slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

# Convert back: a = exp(intercept), b = slope
a = np.exp(intercept)
b = slope
```

**Error Handling:**
- Filter out zero/negative values
- Require minimum 3 positive data points
- Display warning for filtered data

### 3. Logarithmic Regression
```python
# Usage: regression_type="logarithmic"
# Algorithm: Linear regression with log-transformed x
# Equation: y = a*ln(x) + b
# Use Case: Diminishing returns, saturation curves
```

**Mathematical Implementation:**
```python
# Transform x to log space
log_x = np.log(x_data[x_data > 0])
y_filtered = y_data[x_data > 0]

# Linear regression
slope, intercept, r_value, _, _ = stats.linregress(log_x, y_filtered)

# a = slope, b = intercept
```

### 4. Exponential Regression
```python
# Usage: regression_type="exponential"
# Algorithm: Semi-log transformation + linear regression
# Equation: y = ae^(bx) → ln(y) = ln(a) + bx
# Use Case: Growth/decay patterns, customer retention
```

**Mathematical Implementation:**
```python
# Transform y to log space
log_y = np.log(y_data[y_data > 0])
x_filtered = x_data[y_data > 0]

# Linear regression
slope, intercept, r_value, _, _ = stats.linregress(x_filtered, log_y)

# a = exp(intercept), b = slope
a = np.exp(intercept)
b = slope
```

### 5. Polynomial Regression
```python
# Usage: regression_type="polynomial", polynomial_degree=3
# Algorithm: sklearn PolynomialFeatures + LinearRegression
# Equation: y = a₀ + a₁x + a₂x² + ... + aₙxⁿ
# Use Case: Complex curves, seasonal patterns
```

**Mathematical Implementation:**
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create polynomial features
poly_features = PolynomialFeatures(degree=polynomial_degree)
X_poly = poly_features.fit_transform(x_data.reshape(-1, 1))

# Fit regression
reg = LinearRegression().fit(X_poly, y_data)
y_pred = reg.predict(X_poly)
r_squared = r2_score(y_data, y_pred)
```

## 🧮 Equation Display Specifications

### Format Standards
- **Precision**: 3 decimal places for coefficients
- **Scientific Notation**: For values < 0.001 or > 1000
- **Unicode Symbols**: Use proper mathematical notation where supported
- **Fallback**: ASCII equivalents for compatibility

### Equation Templates
```python
equation_formats = {
    "linear": "y = {slope:.3f}x + {intercept:.3f}",
    "power": "y = {a:.3f}x^{b:.3f}",
    "logarithmic": "y = {a:.3f}ln(x) + {b:.3f}",
    "exponential": "y = {a:.3f}e^({b:.3f}x)",
    "polynomial": "y = {coeffs_formatted}"  # Dynamic based on degree
}
```

## 🚨 Error Handling Specifications

### Data Validation

```python
class RegressionValidationError(ValueError):
    """Raised when data is invalid for regression analysis."""
    pass

def validate_regression_data(x_data, y_data, regression_type):
    """Validate data for specific regression type."""
    # Common validations
    if len(x_data) < 2:
        raise RegressionValidationError("At least 2 data points required")

    # Type-specific validations
    if regression_type in ["power", "logarithmic"]:
        positive_x = np.sum(x_data > 0)
        if positive_x < 2:
            raise RegressionValidationError(f"{regression_type} regression requires positive x values")

    if regression_type == "exponential":
        positive_y = np.sum(y_data > 0)
        if positive_y < 2:
            raise RegressionValidationError("Exponential regression requires positive y values")
```

### Fallback Strategy
1. **Primary**: Attempt requested regression type
2. **Warning**: Log data filtering (e.g., removed negative values)
3. **Fallback**: Attempt linear regression if complex type fails
4. **Error**: Raise clear exception if no regression possible

### Error Messages
```python
error_messages = {
    "insufficient_data": "At least {min_points} data points required for {regression_type} regression",
    "invalid_data_type": "{regression_type} regression requires {requirement}",
    "convergence_failed": "{regression_type} regression failed to converge, falling back to linear",
    "no_valid_data": "No valid data points remaining after filtering for {regression_type} regression"
}
```

## 🧪 Testing Specifications

### Test Categories

#### 1. Unit Tests - Algorithm Accuracy
```python
def test_linear_regression_known_data():
    """Test linear regression with known slope/intercept."""
    # Perfect line: y = 2x + 3
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 7, 9, 11, 13])
    # Expected: slope=2.0, intercept=3.0, r²=1.0

def test_power_law_known_data():
    """Test power law with known relationship."""
    # Perfect power law: y = 2x^1.5
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x**1.5
    # Expected: a≈2.0, b≈1.5, r²≈1.0
```

#### 2. Integration Tests - Full Plotting Pipeline
```python
def test_regression_line_on_scatter_plot():
    """Test regression line integration with matplotlib scatter plot."""

def test_regression_line_on_line_plot():
    """Test regression line integration with matplotlib line plot."""

def test_multiple_regression_types_same_data():
    """Test all regression types on same dataset for consistency."""
```

#### 3. Edge Case Tests
```python
def test_negative_values_power_regression():
    """Test power regression with negative x values (should filter)."""

def test_zero_values_logarithmic_regression():
    """Test logarithmic regression with zero x values (should filter)."""

def test_insufficient_data_points():
    """Test behavior with <2 data points."""

def test_identical_data_points():
    """Test behavior with all identical x or y values."""
```

#### 4. Visual Regression Tests
```python
def test_equation_text_positioning():
    """Test equation text doesn't overlap with data or axes."""

def test_equation_formatting():
    """Test equation display format for each regression type."""

def test_r_squared_display():
    """Test R² value display and formatting."""
```

### Test Data Requirements
- **Synthetic datasets** with known parameters for accuracy validation
- **Real retail data** (anonymized) for integration testing
- **Edge case datasets** for robustness testing
- **Performance datasets** (1k, 10k, 100k points) for scalability

## 📚 Documentation Specifications

### Function Docstring Template
```python
def add_regression_line(...) -> Axes:
    """Add a regression line with configurable algorithm to a matplotlib plot.

    Supports multiple regression types for comprehensive trend analysis on retail data.
    All regression types display equations and R² values with customizable formatting.

    Args:
        ax (Axes): Matplotlib axes containing the plot data.
        regression_type (Literal[...], optional): Regression algorithm to use.
            - "linear": y = mx + b (default, OLS regression)
            - "power": y = ax^b (elasticity analysis, log-log transformation)
            - "logarithmic": y = a*ln(x) + b (diminishing returns analysis)
            - "exponential": y = ae^(bx) (growth/decay patterns)
            - "polynomial": y = a₀ + a₁x + ... + aₙxⁿ (complex curves)
            Defaults to "linear".
        [... other parameters ...]

    Returns:
        Axes: The matplotlib axes with regression line and equation added.

    Raises:
        RegressionValidationError: If data is insufficient or invalid for regression type.
        ValueError: If parameters are invalid or plot contains no data.

    Examples:
        Basic linear regression (backward compatible):
        >>> ax = data.plot.scatter(x='price', y='demand')
        >>> gu.add_regression_line(ax)

        Power law for elasticity analysis:
        >>> gu.add_regression_line(ax, regression_type="power")

        Polynomial with custom degree:
        >>> gu.add_regression_line(ax, regression_type="polynomial", polynomial_degree=3)

    Note:
        - Power and logarithmic regression filter out non-positive x values
        - Exponential regression filters out non-positive y values
        - Filtered data warnings are logged but don't raise errors
        - If complex regression fails, automatically falls back to linear
    """
```

### Usage Examples Documentation
```python
# Business Example: Price Elasticity Analysis
ax = price_demand_df.plot.scatter(x='price', y='demand')
gu.add_regression_line(ax, regression_type="power",
                      show_equation=True, show_r2=True)
# Displays: y = 1000.0x^(-1.2), R² = 0.94
# Interpretation: -1.2 elasticity coefficient

# Business Example: Customer Retention Decay
ax = retention_df.plot.scatter(x='days', y='active_customers')
gu.add_regression_line(ax, regression_type="exponential")
# Displays: y = 5000.0e^(-0.05x), R² = 0.87
# Interpretation: 5% daily decay rate
```

## 🚀 Implementation Phases

### Phase 1: Foundation (Priority 1)
- [ ] Add `regression_type` parameter with default "linear"
- [ ] Maintain 100% backward compatibility
- [ ] Enhanced error handling and validation
- [ ] Comprehensive unit tests for linear regression

### Phase 2: Core Algorithms (Priority 2)
- [ ] Implement power, logarithmic, exponential regression
- [ ] Add algorithm-specific data validation
- [ ] Create equation formatting for each type
- [ ] Integration tests with real plotting workflows

### Phase 3: Advanced Features (Priority 3)
- [ ] Polynomial regression with configurable degree
- [ ] Optional confidence interval display
- [ ] Performance optimization for large datasets
- [ ] Advanced error handling and fallbacks

## ✅ Acceptance Criteria

### Functional Requirements
- [ ] All existing `add_regression_line()` calls work unchanged
- [ ] New `regression_type` parameter supports all specified algorithms
- [ ] Equation display adapts to regression type with proper formatting
- [ ] R² calculation accurate for each algorithm type
- [ ] Data validation prevents crashes on edge cases

### Quality Requirements
- [ ] Test coverage ≥95% for new regression functionality
- [ ] Performance acceptable for datasets up to 10k points
- [ ] Error messages clear and actionable for users
- [ ] Documentation includes business-focused examples
- [ ] Visual regression tests prevent UI regressions

### Compatibility Requirements
- [ ] No breaking changes to existing API
- [ ] All current test cases continue to pass
- [ ] New optional parameters have sensible defaults
- [ ] Function behavior identical for `regression_type="linear"`

---

**Next Step**: Review and approve this specification before beginning implementation.