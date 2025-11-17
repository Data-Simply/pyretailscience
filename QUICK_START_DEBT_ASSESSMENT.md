# Quick Start: Technical Debt Assessment

## TL;DR - Run This Now

```bash
# Install required tools (one-time)
pip install pylint radon vulture

# Run the assessment
./scripts/assess_technical_debt.sh

# Review the output
cat technical_debt_report.txt

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## What You Get

The assessment automatically scans for:

1. **Code Duplication** - Copy-pasted code violating DRY
2. **Complexity** - Functions that are too complex (>10 cyclomatic complexity)
3. **Dead Code** - Unused functions, imports, variables
4. **Deprecated Patterns** - Old APIs, TODO markers
5. **Test Coverage** - Gaps in test coverage
6. **Long Functions** - Functions >50 lines
7. **Deep Nesting** - Code with >4 indentation levels

## Interpreting Results

### Priority Matrix

```
High Impact + Low Effort = QUICK WIN ⭐ (Do immediately)
High Impact + High Effort = BIG BET 🎯 (Plan for sprint)
Low Impact + Low Effort = FILL-IN 🔧 (Do when idle)
Low Impact + High Effort = TIME SINK ⏰ (Skip/defer)
```

### Scoring Guide

Use this formula to prioritize:
```
Score = (Impact × 3 + Knowledge × 2) - (Effort × 2)

Impact:     Critical=4, High=3, Medium=2, Low=1
Knowledge:  High=3, Medium=2, Low=1 (how well you know this code)
Effort:     Very High=4, High=3, Medium=2, Low=1
```

Higher score = higher priority

### Example Decision Making

**Finding: Duplicate validation code in 3 files (45 lines)**
- Impact: High (maintenance burden, bug risk)
- Knowledge: High (you wrote it)
- Effort: Low (1-2 hours to extract to function)
- **Score: (3×3 + 3×2) - (1×2) = 13** ⭐ QUICK WIN

**Finding: Refactor entire plotting architecture**
- Impact: Medium (improves code quality)
- Knowledge: Low (complex, unfamiliar)
- Effort: Very High (5+ days)
- **Score: (2×3 + 1×2) - (4×2) = 0** ⏰ TIME SINK (defer)

## First Week Action Plan

1. **Day 1: Run Assessment**
   ```bash
   ./scripts/assess_technical_debt.sh
   ```

2. **Day 2: Review & Categorize**
   - Read `technical_debt_report.txt`
   - Identify top 3 Quick Wins
   - Create GitHub issues for each

3. **Day 3-4: Execute Quick Wins**
   - Fix 1-2 Quick Wins
   - Run tests to verify
   - Commit with descriptive messages

4. **Day 5: Review & Plan**
   - Measure impact (lines removed, complexity reduced)
   - Identify next week's Quick Wins
   - Plan one Big Bet for sprint

## Common Quick Wins to Look For

### 1. Extract Magic Numbers
```python
# Before
if sales > 1000:  # What does 1000 mean?

# After
PREMIUM_CUSTOMER_THRESHOLD = 1000
if sales > PREMIUM_CUSTOMER_THRESHOLD:
```

### 2. Replace Deprecated pandas
```python
# Before
df.ix[5]  # Deprecated
df = df.append(new_row)  # Deprecated

# After
df.iloc[5]  # or df.loc[5]
df = pd.concat([df, pd.DataFrame([new_row])])
```

### 3. Extract Repeated Validation
```python
# Before (repeated in 5 functions)
if df is None:
    raise ValueError("DataFrame cannot be None")
if not isinstance(df, pd.DataFrame):
    raise TypeError("Expected DataFrame")

# After (extract to utils/validation.py)
def validate_dataframe(df: pd.DataFrame, name: str = "df") -> None:
    if df is None:
        raise ValueError(f"{name} cannot be None")
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
```

### 4. Simplify Nested Conditionals
```python
# Before
def process_customer(customer):
    if customer is not None:
        if customer.active:
            if customer.sales > 0:
                return calculate_score(customer)
            else:
                return 0
        else:
            return None
    else:
        raise ValueError("Customer required")

# After (guard clauses)
def process_customer(customer):
    if customer is None:
        raise ValueError("Customer required")
    if not customer.active:
        return None
    if customer.sales <= 0:
        return 0
    return calculate_score(customer)
```

### 5. Vectorize DataFrame Operations
```python
# Before (slow)
results = []
for idx, row in df.iterrows():
    if row['sales'] > 100:
        results.append(row['customer_id'])

# After (vectorized)
results = df.loc[df['sales'] > 100, 'customer_id'].tolist()
```

## Tracking Progress

### Option 1: GitHub Issues

Create issues with this template:

```markdown
## [Quick Win] Extract Repeated Validation Logic

**Files:** utils/label.py, utils/filter_and_label.py, analysis/customer.py
**Lines:** ~45 lines of duplication
**Effort:** 1-2 hours
**Impact:** High (reduces bugs, improves maintainability)

### Current State
DataFrame validation is duplicated across 3 files...

### Proposed Solution
Create `utils/validation.py` with shared validation functions

### Acceptance Criteria
- [ ] validation.py created with validate_dataframe()
- [ ] All 3 files updated to use shared function
- [ ] Tests added for validation.py
- [ ] No reduction in coverage
```

### Option 2: Simple Spreadsheet

| ID | Category | Description | File | Effort | Impact | Status |
|----|----------|-------------|------|--------|--------|--------|
| 1 | DRY | Extract validation | utils/label.py | Low | High | In Progress |
| 2 | DEPRECATION | Replace .ix | analysis/customer.py | Low | Critical | Open |

## Monthly Metrics to Track

Run assessment monthly and track trends:

```bash
# Create monthly snapshot
./scripts/assess_technical_debt.sh
cp technical_debt_report.txt reports/debt_report_$(date +%Y-%m).txt
```

Track these KPIs:
- **Duplication %** - Target: <5%
- **Avg Complexity** - Target: ≤5
- **Dead Code Items** - Target: 0
- **Test Coverage** - Target: >85%
- **Ruff Violations** - Target: 0

## Resources

- **Full Framework:** `TECHNICAL_DEBT_FRAMEWORK.md` (comprehensive guide)
- **Assessment Script:** `scripts/assess_technical_debt.sh` (run anytime)
- **Latest Report:** `technical_debt_report.txt` (generated by script)
- **Coverage Report:** `htmlcov/index.html` (visual coverage gaps)

## Questions?

Refer to specific sections in `TECHNICAL_DEBT_FRAMEWORK.md`:
- **Section 1:** Detailed category definitions
- **Section 2:** Effort vs Impact matrix
- **Section 3:** Full assessment process
- **Section 7:** PyRetailScience-specific patterns

---

**Remember:** Start small, measure impact, build momentum. Even 2-3 Quick Wins per week compounds into major improvements over months!
