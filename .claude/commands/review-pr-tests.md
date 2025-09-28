# Input Validation
if [ -z "$1" ]; then
    echo "Error: PR number is required. Usage: /review-pr-tests <PR_NUMBER>"
    exit 1
fi

You are reviewing all Python tests in PR #$1 for the PyRetailScience package.

## Initial Analysis Steps:
1. **First, check the PR using `gh pr view $1` and `gh pr diff $1`** to see the changes
2. **If you need to examine test files locally**, check out the PR:
   ```bash
   # This automatically fetches and checks out the PR's branch
   # (handles both local and remote branches)
   gh pr checkout $1
   ```
3. **Check if there are any Python test files** in the PR (files matching `test_*.py` or `*_test.py`). If no test files exist, provide a brief note about missing tests
4. **Identify which PyRetailScience module is being tested** by analyzing the imports and function calls in the code

If there are no test files in the PR, respond with:
```
## NO TEST FILES FOUND IN PR #$1

This PR does not contain any test files. Consider adding tests for:
- [List key functionality that should be tested based on the PR changes]
```

Otherwise, identify tests that fall into the following problematic categories:

## 1. Tests That Don't Test Package Code
Tests that make assertions but never call any functions, classes, or modules from the package under test. Look for tests that only test local variables, Python built-ins, or standard library functions without involving the actual package.

**Red Flags:**
- No imports from the package being tested
- Only tests variable assignments or basic Python operations
- Never calls functions/methods from the package

**Example:**
```python
def test_basic_math():
    # BAD: Never uses the package under test
    x = 1
    y = 2
    assert x + y == 3

def test_string_operations():
    # BAD: Only tests Python built-in string methods
    name = "John"
    assert name.upper() == "JOHN"
    assert len(name) == 4
```

## 2. Tests That Primarily Test Library Functionality
Tests that focus on verifying external library behavior rather than how the package under test uses those libraries. These tests essentially re-test third-party libraries that should already have their own test suites.

**Red Flags:**
- Directly calls library functions without package wrapper
- Tests properties of library outputs (e.g., UUID format, datetime formatting)
- Could be moved to the library's own test suite without loss

**Example:**
```python
import uuid
import pandas as pd
from datetime import datetime

def test_uuid_generation():
    # BAD: Tests uuid library, not the package's use of it
    generated_id = str(uuid.uuid4())
    assert len(generated_id) == 36
    assert generated_id.count("-") == 4

def test_pandas_dataframe():
    # BAD: Tests pandas functionality, not package code
    df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    assert len(df) == 2
    assert 'A' in df.columns

# GOOD: Tests package's use of the library
def test_crossshop_analysis():
    from pyretailscience.crossshop import CrossShop
    analyzer = CrossShop(store_col="store_id")
    result = analyzer.fit(df)
    assert result.shape[0] > 0  # Tests that package processes data
    assert "customer_id" in result.columns  # Tests package output
```

## 3. Tests With Substantial Duplication
Tests that repeat significant portions of logic, assertions, or setup from other tests without adding meaningful coverage. Look for copy-pasted test code with minor variations that don't test different behavior.

**Red Flags:**
- Multiple tests with nearly identical setup code
- Same assertions repeated across tests with minor value changes
- Tests that differ only in input values but test the same code path
- Could be combined using parametrize or merged into one test

**Example:**
```python
def test_add_positive_numbers():
    calc = Calculator()
    result = calc.add(5, 3)
    assert result == 8
    assert result > 0
    assert isinstance(result, int)

def test_add_different_positive_numbers():
    # BAD: Duplicates above test with different values
    calc = Calculator()
    result = calc.add(10, 7)
    assert result == 17
    assert result > 0
    assert isinstance(result, int)

def test_add_more_numbers():
    # BAD: Same pattern again
    calc = Calculator()
    result = calc.add(2, 4)
    assert result == 6
    assert result > 0
    assert isinstance(result, int)

# BETTER: Use parametrize or combine into one comprehensive test
@pytest.mark.parametrize("a,b,expected", [
    (5, 3, 8),
    (10, 7, 17),
    (2, 4, 6)
])
def test_add_positive_numbers(a, b, expected):
    calc = Calculator()
    result = calc.add(a, b)
    assert result == expected
    assert result > 0
    assert isinstance(result, int)
```

## 4. Tests That Cover Only Basic Language/Library Features
Tests that verify fundamental Python or standard library behavior without meaningful connection to PyRetailScience's functionality. These tests essentially verify that Python works as expected.

**Red Flags:**
- Tests basic Python operations (list, dict, set operations)
- Verifies standard library behavior that's guaranteed by Python
- No connection to business logic or package functionality
- Could be in Python's own test suite

**Example:**
```python
def test_basic_set_operations():
    # BAD: Tests Python's set behavior
    assert len(set(["Hello", "World"])) == 2
    assert "Hello" in set(["Hello", "World"])

def test_dictionary_basics():
    # BAD: Tests Python dict functionality
    d = {"key": "value"}
    assert d.get("key") == "value"
    assert d.get("missing", "default") == "default"

def test_list_operations():
    # BAD: Tests Python list methods
    items = [1, 2, 3]
    items.append(4)
    assert len(items) == 4
    assert items[-1] == 4
```

## Additional Problematic Patterns to Identify:

### 5. Trivial or Tautological Tests
Tests that assert obvious truths, constants equal themselves, or test definitions rather than behavior. These provide no value and false confidence in coverage metrics.

**Red Flags:**
- Tests that a constant equals its defined value
- Assertions that are always true by definition
- Tests that verify class/function names or attributes exist
- No actual behavior or logic being tested

**Example:**
```python
# Assuming MAX_STORES = 100 in pyretailscience.config
def test_constant():
    from pyretailscience.config import MAX_STORES
    # BAD: Just tests that 100 == 100
    assert MAX_STORES == 100

def test_true_is_true():
    # BAD: Tautological
    assert True == True
    assert 1 == 1

def test_class_name():
    from pyretailscience.style import BaseStyle
    # BAD: Tests Python's class system, not package behavior
    obj = BaseStyle()
    assert obj.__class__.__name__ == "BaseStyle"

def test_config_values():
    # BAD if CONFIG is just {"debug": False}
    from pyretailscience import CONFIG
    assert CONFIG["debug"] == False  # Just repeating the definition
```

### 6. Tests With No Meaningful Assertions
Tests that either have no assertions, only assert True, or have assertions that can never fail. These tests may execute code but don't verify behavior or outcomes.

**Red Flags:**
- No assert statements at all
- Only checks if attributes/methods exist
- Assertions that can never fail given the test setup
- Tests that only verify successful execution without checking results

**Example:**
```python
def test_function_runs():
    from pyretailscience.segmentation import RFM
    # BAD: No assertions
    rfm = RFM()
    rfm.fit(df)
    # Test passes even if function does nothing or fails silently

def test_function_exists():
    import pyretailscience.style
    # BAD: Only tests existence, not behavior
    assert hasattr(pyretailscience.style, 'apply_style')
    assert callable(pyretailscience.style.apply_style)

def test_always_passes():
    from pyretailscience.customer_choice import ProductAssociation
    # BAD: Assertion can never fail
    pa = ProductAssociation()
    confidence = pa.get_confidence()
    assert confidence >= 0 or confidence < 0  # Always true for any number

def test_object_creation():
    from pyretailscience.pricing import PriceElasticity
    # BAD: Only tests that no exception is raised
    obj = PriceElasticity()
    assert obj is not None  # Objects are never None after creation

# GOOD: Test actual behavior
def test_rfm_segmentation():
    from pyretailscience.segmentation import RFM
    rfm = RFM()
    result = rfm.fit_predict(df)
    assert 'segment' in result.columns  # Verifies actual output
    assert len(result) == len(df)  # Verifies processing
```

### 7. Over-Mocked Tests
Tests that mock so extensively that they don't actually test real behavior or integration. When everything is mocked, you're only testing that mocks return what you told them to return.

**Red Flags:**
- Mocks every external dependency AND internal components
- Test would pass even if the actual implementation is completely broken
- Mocks return values that are directly asserted
- No real code execution happening
- More mock setup than actual test code

**Example:**
```python
def test_crossshop_analysis(mocker):
    from pyretailscience.crossshop import CrossShop
    # BAD: Mocks everything, tests nothing real
    mocker.patch('pyretailscience.crossshop._validate_data', return_value=True)
    mocker.patch('pyretailscience.crossshop._calculate_matrix', return_value=pd.DataFrame())
    mocker.patch('pyretailscience.crossshop._apply_filters', return_value=pd.DataFrame())
    mocker.patch('pyretailscience.crossshop._format_output', return_value={'result': 'done'})

    cs = CrossShop()
    result = cs.fit(df)
    assert result == {'result': 'done'}  # Only tests that mock returns 'done'

def test_pricing_optimization(mocker):
    from pyretailscience.pricing import optimize_prices
    # BAD: Nothing real is tested
    mocker.patch('pyretailscience.pricing.validate_input', return_value=True)
    mocker.patch('pyretailscience.pricing.calculate_elasticity', return_value=1.5)
    mocker.patch('pyretailscience.pricing.find_optimal', return_value=99.99)
    mocker.patch('pyretailscience.pricing.apply_constraints', return_value=95.00)

    result = optimize_prices(product_data)
    assert result == 95.00  # Just testing mock configuration

# GOOD: Mock external dependencies but test real logic
def test_rfm_with_real_calculation(mocker):
    from pyretailscience.segmentation import RFM
    # Mock only data loading, not the actual RFM logic
    mocker.patch('pyretailscience.segmentation.load_data', return_value=test_df)

    # Test real RFM calculation logic
    rfm = RFM()
    result = rfm.calculate_scores(test_df)
    assert 'R_score' in result.columns  # Tests real scoring logic
```

### 8. Tests That Verify Implementation Details
Tests that check internal data structures, state, or "how" something works rather than "what" it produces. These tests break when refactoring even if the public behavior remains correct. Note: Testing private method BEHAVIOR (outputs) is often acceptable for complex algorithms.

**Red Flags:**
- Tests internal data structures or formats
- Verifies private attributes that track internal state
- Checks exact method call sequences or execution paths
- Tests "how" rather than "what"
- Would break if you change implementation but keep same behavior

**Key Distinction:**
- **BAD**: Testing internal state/structure → `assert obj._cache == {...}`
- **OK**: Testing private method behavior → `assert obj._calculate(5) == 10`

**Example:**
```python
def test_internal_cache_structure():
    from pyretailscience.style import StyleCache
    # BAD: Tests internal data structure/state
    cache = StyleCache()
    cache.add_style("retail", {"color": "blue"})

    # Testing HOW data is stored internally
    assert isinstance(cache._styles, dict)
    assert len(cache._cache_keys) == 1
    assert cache._counter == 1  # Private state tracking
    assert cache._styles['retail']['format'] == 'processed'

def test_rfm_algorithm_internals():
    from pyretailscience.segmentation import RFM
    # BAD: Tests HOW algorithm works internally
    rfm = RFM()
    rfm.fit(customer_data)

    # Testing internal algorithm mechanics
    assert rfm._percentile_breaks == [25, 50, 75]
    assert rfm._score_mappings == {1: 'low', 2: 'med', 3: 'high'}
    assert rfm._calculation_order == ['R', 'F', 'M']

def test_internal_validation_calls(mocker):
    from pyretailscience.crossshop import CrossShop
    # BAD: Tests exact execution path
    cs = CrossShop()
    spy = mocker.spy(cs, '_validate_columns')

    cs.fit(df)
    spy.assert_called_once_with(df.columns)  # Tests call sequence

# GOOD: Test behavior/output, even of private methods
def test_elasticity_calculation():
    from pyretailscience.pricing import PriceElasticity
    # GOOD: Testing WHAT a private method produces
    pe = PriceElasticity(base_price=100)
    elasticity = pe._calculate_point_elasticity(95, 1000, 105, 900)
    assert abs(elasticity + 2.0) < 0.01  # Tests mathematical correctness

def test_crossshop_matrix_output():
    from pyretailscience.crossshop import CrossShop
    # GOOD: Tests public behavior
    cs = CrossShop(store_col="store_id")
    result = cs.create_matrix(transaction_data)
    assert result.shape == (n_stores, n_stores)  # Tests output structure
```

### 9. Tests That Could Use Parametrization
Multiple test functions that test the same behavior with different inputs but don't use pytest's parametrization feature. This leads to code duplication and harder maintenance.

**Use pytest.mark.parametrize for:**
- Tests with identical structure but different input values
- Multiple test cases that differ only in data
- Boundary value testing with multiple edge cases

**Red Flags:**
- Multiple test functions with identical structure
- Only differences are input values and expected outputs
- Same setup and assertion patterns
- Test names differ only by the values being tested
- Could be a single parametrized test

**Example:**
```python
# BAD: Repetitive test functions
def test_divide_by_two():
    assert calculator.divide(10, 2) == 5

def test_divide_by_five():
    assert calculator.divide(20, 5) == 4

def test_divide_by_ten():
    assert calculator.divide(100, 10) == 10

def test_divide_negative():
    assert calculator.divide(-10, 2) == -5

def test_validate_email_gmail():
    assert validator.is_valid("user@gmail.com") == True

def test_validate_email_yahoo():
    assert validator.is_valid("user@yahoo.com") == True

def test_validate_email_invalid():
    assert validator.is_valid("not-an-email") == False

# GOOD: Use parametrize
@pytest.mark.parametrize("dividend,divisor,expected", [
    (10, 2, 5),
    (20, 5, 4),
    (100, 10, 10),
    (-10, 2, -5),
])
def test_divide(dividend, divisor, expected):
    assert calculator.divide(dividend, divisor) == expected

@pytest.mark.parametrize("email,is_valid", [
    ("user@gmail.com", True),
    ("user@yahoo.com", True),
    ("not-an-email", False),
])
def test_validate_email(email, is_valid):
    assert validator.is_valid(email) == is_valid
```

### 10. Tests With Generic or Non-Descriptive Test Data
Tests that use generic placeholder values (like "A", "B", "C", "test", "data", "123") instead of realistic, domain-specific examples that make the test's purpose and behavior clearer.

**Example:**
```python
def test_rfm_segmentation():
    # Generic, non-descriptive test data
    customers = ["A", "B", "C"]
    transactions = [
        {"customer": "test", "amount": 100, "date": "2024-01-01"},
        {"customer": "data", "amount": 200, "date": "2024-01-02"}
    ]

    rfm = RFM()
    result = rfm.fit_predict(transactions)
    assert "segment" in result.columns

# Better with realistic retail data:
def test_rfm_segmentation():
    # Realistic retail customer data
    transactions = [
        {"customer_id": "CUST_10234", "amount": 156.99, "date": "2024-01-15", "store": "Store_001"},
        {"customer_id": "CUST_10235", "amount": 89.50, "date": "2024-01-16", "store": "Store_002"},
        {"customer_id": "CUST_10234", "amount": 234.75, "date": "2024-02-01", "store": "Store_001"}
    ]

    rfm = RFM()
    result = rfm.fit_predict(pd.DataFrame(transactions))
    assert "segment" in result.columns
```

**Why this matters:**
- Realistic data makes tests easier to understand and relate to actual use cases
- Helps developers quickly grasp what the test is validating
- Makes debugging easier when tests fail
- Serves as implicit documentation of expected data formats and business logic
- Reduces cognitive load when reading and maintaining tests

### 11. Tests That Don't Verify Their Claimed Behavior
Tests where the function name, docstring, or description claims to test specific behavior, but the assertions are too generic to actually verify that behavior. These tests create false confidence by appearing to test important functionality while only checking trivial outcomes.

**Red Flags:**
- Test name mentions specific behavior (e.g., "sorts", "validates", "filters", "transforms")
- Docstring describes particular functionality being tested
- Assertions only check return types, basic existence, or that no exceptions occurred
- Missing assertions that would actually verify the claimed behavior
- Could pass even if the specific behavior is completely broken

**Example:**
```python
def test_plot_with_unsorted_bins_list():
    """Test price architecture plot automatically sorts unsorted bins list."""
    df = pd.DataFrame({
        "unit_price": [1, 2, 3],
        "retailer": ["Walmart", "Target", "Amazon"],
    })

    # BAD: Claims to test sorting but doesn't verify it
    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[3, 1, 2],  # Unsorted bins - but test doesn't verify they get sorted!
    )

    assert isinstance(result_ax, Axes)  # Only checks return type

def test_validates_email_format():
    """Test that email validation properly checks format."""
    validator = EmailValidator()

    # BAD: Claims to test validation but doesn't verify it
    result = validator.validate("not-an-email")
    assert result is not None  # Doesn't check if validation actually worked

def test_filters_inactive_customers():
    """Test customer filtering removes inactive customers."""
    customers = create_test_customers()  # Mix of active/inactive

    # BAD: Claims to test filtering but doesn't verify the filtering logic
    filtered = CustomerFilter().filter_active(customers)
    assert len(filtered) >= 0  # Trivial assertion, doesn't verify filtering

def test_calculates_price_elasticity():
    """Test price elasticity calculation returns correct values."""
    price_data = [100, 110, 120, 130]
    demand_data = [1000, 900, 800, 700]

    # BAD: Claims to test calculation but doesn't verify correctness
    elasticity = calculate_elasticity(price_data, demand_data)
    assert elasticity is not None  # Doesn't verify the calculation is correct

# GOOD: Actually verify the claimed behavior
def test_plot_with_unsorted_bins_list():
    """Test price architecture plot automatically sorts unsorted bins list."""
    df = pd.DataFrame({
        "unit_price": [1, 2, 3],
        "retailer": ["Walmart", "Target", "Amazon"],
    })

    result_ax = price.plot(
        df=df,
        value_col="unit_price",
        group_col="retailer",
        bins=[3, 1, 2],  # Unsorted bins
    )

    # GOOD: Verify the bins were actually sorted by checking output
    y_labels = [label.get_text() for label in result_ax.get_yticklabels()]
    assert "$1.0 - $2.0" in y_labels[0]  # Sorted order: first bin is 1-2
    assert "$2.0 - $3.0" in y_labels[1]  # Second bin is 2-3

def test_validates_email_format():
    """Test that email validation properly checks format."""
    validator = EmailValidator()

    # GOOD: Actually test validation behavior
    assert validator.validate("user@example.com") == True
    assert validator.validate("invalid-email") == False
    assert validator.validate("missing@") == False

def test_filters_inactive_customers():
    """Test customer filtering removes inactive customers."""
    active_customers = [Customer(id=1, active=True), Customer(id=2, active=True)]
    inactive_customers = [Customer(id=3, active=False), Customer(id=4, active=False)]
    all_customers = active_customers + inactive_customers

    # GOOD: Verify filtering actually works
    filtered = CustomerFilter().filter_active(all_customers)
    assert len(filtered) == 2  # Should have exactly 2 active customers
    assert all(c.active for c in filtered)  # All should be active
    assert {c.id for c in filtered} == {1, 2}  # Should be the right customers
```

**Why this matters:**
- Test names and docstrings serve as documentation of what functionality is tested
- False confidence is dangerous - tests appear to cover behavior but actually don't
- Bugs in the claimed functionality will go undetected
- Future developers may assume the behavior is tested and working correctly
- Makes debugging harder when the "tested" functionality actually fails

## Instructions:
1. **View the PR** using `gh pr view $1` and `gh pr diff $1` to understand the changes
2. **Optionally checkout the PR** if local examination is needed using `gh pr checkout $1` (this fetches if necessary)
3. **Check for Python test files** - if none exist, provide guidance on what should be tested based on the PR changes
4. **Identify the PyRetailScience module under test** by analyzing imports and function calls in the test code
5. Review each test function in the provided PR
6. Categorize any problematic tests using the categories above (1-11)
7. **Highlight the specific problematic code** from each test function
8. Provide brief explanations for why each flagged test is problematic
9. Suggest improvements where appropriate (e.g., "combine with test_X", "use pytest.parametrize", "add actual package functionality", "use realistic retail/business data")
10. **When suggestions involve simple fixes, provide the corrected code**
11. Highlight any tests that might be salvageable with minor modifications

## Output Format:
```
## LANGUAGE: Python
## PACKAGE UNDER TEST: pyretailscience.[module_name]

PROBLEMATIC TESTS IDENTIFIED:

### Test: test_function_name()

**Issue**: [Brief description] [Lines affected]

Problematic code: ```python
[highlight the specific problematic code from the test]
```

**Suggestion**: [How to improve or whether to remove]

Code fix: ```python
[if suggestion includes a simple fix, show the corrected code here]
```

## POSITIVE OBSERVATIONS

The following tests are well-structured and properly test the package functionality:

1. test_function_name (test code lines) - very brief description of the test

SUMMARY:
- Language: Python
- Package analyzed: pyretailscience.[module_name]
- Total tests reviewed: [X]
- Problematic tests found: [X]

## Recommendations

[A numbered list of recommendations - if any]
```
