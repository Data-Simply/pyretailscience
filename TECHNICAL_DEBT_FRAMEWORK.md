# Technical Debt Assessment Framework for PyRetailScience

## Executive Summary

This framework provides a systematic approach to identify, categorize, and prioritize code quality improvements across deprecation, DRY violations, and simplification opportunities. It combines industry-standard methodologies with Python-specific tooling to deliver actionable insights with effort-vs-impact scoring.

**Target Metrics:**
- Code duplication: <5% (industry standard)
- Cyclomatic complexity: ≤10 per function (already configured in ruff)
- Deprecated code: 0 instances
- Test coverage: Maintain >80%

---

## 1. Technical Debt Categories

### 1.1 Deprecated Code (DEPRECATION)
**Definition:** Code that should no longer be used, often replaced by better implementations.

**Detection Methods:**
```bash
# Find deprecated decorators and warnings
rg "DeprecationWarning|deprecated|@deprecated" --type py

# Find TODO/FIXME/XXX comments indicating planned deprecation
rg "TODO.*deprecat|FIXME.*deprecat|XXX.*deprecat" --type py -i

# Find old API patterns (manual review needed)
rg "\.ix\[|\.append\(.*dict|inplace=True" pyretailscience/

# Check for usage of deprecated pandas/numpy/sklearn methods
grep -r "\.ix\[" pyretailscience/  # Deprecated pandas indexer
grep -r "np\.bool\b" pyretailscience/  # Deprecated numpy type
```

**Effort Scoring:**
- **Low (1-2 hours):** Single function/method replacement
- **Medium (2-8 hours):** Multiple files, some refactoring needed
- **High (1-3 days):** API changes requiring downstream updates

**Impact Scoring:**
- **Critical:** Blocks upgrades, security issues, breaks in new Python versions
- **High:** Performance impact, maintainability issues
- **Medium:** Code smell, confusion for contributors
- **Low:** Cosmetic, no functional impact

---

### 1.2 DRY Violations (Code Duplication)

**Definition:** Code duplication violating "Don't Repeat Yourself" principle.

**Detection Methods:**
```bash
# Install and run code duplication detector
pip install pylint
pylint --disable=all --enable=duplicate-code pyretailscience/

# Alternative: use jscpd (language-agnostic)
npx jscpd pyretailscience/ --min-lines 5 --min-tokens 50 --format python

# Manual pattern search for common duplications
rg "def.*\(df.*DataFrame" pyretailscience/ -A 20 | less  # Similar function signatures
rg "if.*is not None" pyretailscience/ -A 3  # Repeated validation patterns
rg "\.groupby|\.agg|\.transform" pyretailscience/ -A 5  # Similar pandas operations
```

**Categorization:**
- **Type 1:** Exact clones (copy-paste)
- **Type 2:** Parameterized clones (same structure, different identifiers)
- **Type 3:** Gapped clones (similar with modifications)
- **Type 4:** Semantic clones (same behavior, different implementation)

**Effort Scoring:**
- **Low (1-4 hours):** Extract to shared function/constant
- **Medium (4-12 hours):** Refactor to base class or utility module
- **High (1-5 days):** Requires architectural changes

**Impact Scoring:**
- **Critical:** Bug in duplicated code affects multiple modules
- **High:** >10 instances, maintenance burden
- **Medium:** 3-10 instances
- **Low:** 2 instances, low-change areas

---

### 1.3 Complexity & Simplification (COMPLEXITY)

**Definition:** Overly complex code that could be simplified without changing behavior.

**Detection Methods:**
```bash
# Cyclomatic complexity (already in ruff config: max 10)
uv run ruff check . --select C901  # McCabe complexity

# Cognitive complexity (more nuanced than cyclomatic)
pip install radon
radon cc pyretailscience/ -a -nb  # Average complexity, no bar chart

# Deep nesting detection
rg "^\s{16,}" pyretailscience/ --type py  # 4+ levels of indentation

# Long functions (>50 lines often indicate complexity)
find pyretailscience/ -name "*.py" -exec awk '/^def |^    def /{start=NR; name=$0} /^def |^    def |^class /{if(NR-start>50 && name) print FILENAME":"start":"name} END{if(NR-start>50 && name) print FILENAME":"start":"name}' {} +

# Long parameter lists (>5 params)
rg "def \w+\([^)]*,.*,.*,.*,.*," pyretailscience/ --type py

# God classes (>500 lines, >20 methods)
find pyretailscience/ -name "*.py" -exec wc -l {} + | awk '$1 > 500'
```

**Opportunity Types:**
- **Extract Method:** Long functions → smaller focused functions
- **Extract Class:** God classes → cohesive classes
- **Replace Magic Numbers:** Hardcoded values → named constants
- **Simplify Conditionals:** Nested ifs → guard clauses, polymorphism
- **Vectorization:** Loops on DataFrames → pandas/numpy operations
- **Type Hints:** Missing annotations → explicit types

**Effort Scoring:**
- **Low (1-3 hours):** Extract constants, simple refactoring
- **Medium (3-10 hours):** Extract methods, simplify conditionals
- **High (2-5 days):** Extract classes, major restructuring

**Impact Scoring:**
- **Critical:** Performance bottleneck, frequently changed
- **High:** Hard to understand, blocks new features
- **Medium:** Moderately complex, low change frequency
- **Low:** Complex but stable, rarely touched

---

### 1.4 Dead Code (UNUSED)

**Definition:** Code that is never executed or referenced.

**Detection Methods:**
```bash
# Use vulture to find dead code
pip install vulture
vulture pyretailscience/ --min-confidence 80

# Find unused imports (ruff already catches this)
uv run ruff check . --select F401

# Find private methods never called
rg "^\s+def _\w+" pyretailscience/ --type py -o | sort | uniq -c

# Cross-reference with grep to find usage
for func in $(rg "^def _\w+" pyretailscience/ -oNI | cut -d: -f2); do
    count=$(rg "$func" pyretailscience/ | wc -l)
    if [ $count -eq 1 ]; then
        echo "Potentially unused: $func"
    fi
done
```

**Effort Scoring:**
- **Low (30min-1 hour):** Delete unused imports, variables
- **Medium (1-3 hours):** Remove unused methods, verify no external usage
- **High (3-8 hours):** Remove unused classes, modules

**Impact Scoring:**
- **High:** Confuses developers, increases cognitive load
- **Medium:** Adds maintenance burden
- **Low:** Minimal code, low confusion risk

---

### 1.5 Test Quality Issues (TEST-DEBT)

**Definition:** Missing tests, low coverage, or poor test quality.

**Detection Methods:**
```bash
# Coverage analysis
uv run pytest --cov=pyretailscience --cov-report=term-missing --cov-report=html

# Find untested public functions
# 1. Get all public functions
rg "^def [^_]" pyretailscience/ --type py -o -N | cut -d: -f2 > /tmp/public_funcs.txt
# 2. Check if tests exist
while read func; do
    if ! rg "$func" tests/ -q; then
        echo "Untested: $func"
    fi
done < /tmp/public_funcs.txt

# Test quality checks
rg "assert.*is not None" tests/  # Weak assertions
rg "pass$" tests/ -A -1 -B 1  # Empty tests
rg "# TODO|# FIXME|pytest.skip" tests/  # Skipped/incomplete tests
```

**Effort Scoring:**
- **Low (1-2 hours):** Add tests for simple functions
- **Medium (2-6 hours):** Integration tests, complex scenarios
- **High (1-3 days):** Test infrastructure, mocking setup

**Impact Scoring:**
- **Critical:** No tests for critical business logic
- **High:** <60% coverage, core modules untested
- **Medium:** 60-80% coverage, edge cases missing
- **Low:** >80% coverage, minor gaps

---

## 2. Effort vs Impact Matrix

Use this matrix to prioritize work:

```
                        IMPACT
                Low     Medium    High      Critical
        ┌───────────────────────────────────────────┐
    Low │ Skip    Fill-in   Quick Win  Quick Win    │
        │                                           │
 E Medium│ Skip    Fill-in   Quick Win  Big Bet     │
 F      │                                           │
 F High │ Skip    Skip      Big Bet    Big Bet      │
 O      │                                           │
 R Very │ Skip    Skip      Time Sink  Big Bet      │
 T High │                                           │
        └───────────────────────────────────────────┘

Priority Order: Quick Wins > Big Bets > Fill-ins > Skip/Time Sinks
```

**Quadrant Definitions:**
- **Quick Wins:** Do immediately (high impact, low effort)
- **Big Bets:** Schedule for sprint (high impact, high effort)
- **Fill-ins:** Do when waiting between tasks (low impact, low effort)
- **Time Sinks:** Avoid or defer indefinitely (low impact, high effort)

---

## 3. Comprehensive Assessment Process

### Phase 1: Automated Discovery (2-3 hours)

Run all automated tools and collect data:

```bash
#!/bin/bash
# Save as: scripts/assess_technical_debt.sh

echo "=== Technical Debt Assessment Report ===" > debt_report.txt
echo "Generated: $(date)" >> debt_report.txt
echo "" >> debt_report.txt

# 1. Code duplication
echo "## Code Duplication" >> debt_report.txt
pylint --disable=all --enable=duplicate-code pyretailscience/ 2>&1 | tee -a debt_report.txt

# 2. Complexity analysis
echo -e "\n## Complexity Analysis" >> debt_report.txt
radon cc pyretailscience/ -a -s | tee -a debt_report.txt

# 3. Dead code
echo -e "\n## Dead Code" >> debt_report.txt
vulture pyretailscience/ --min-confidence 80 | tee -a debt_report.txt

# 4. Linting issues
echo -e "\n## Linting Issues" >> debt_report.txt
uv run ruff check . --statistics | tee -a debt_report.txt

# 5. Coverage gaps
echo -e "\n## Coverage Analysis" >> debt_report.txt
uv run pytest --cov=pyretailscience --cov-report=term-missing | tee -a debt_report.txt

# 6. Deprecated patterns (manual review needed)
echo -e "\n## Potential Deprecations" >> debt_report.txt
rg "TODO.*deprecat|FIXME|XXX|HACK" pyretailscience/ -n | tee -a debt_report.txt

echo -e "\nReport saved to: debt_report.txt"
```

### Phase 2: Manual Review (3-5 hours)

Review code for patterns automation misses:

1. **Architectural smells:** Circular dependencies, tight coupling
2. **API inconsistencies:** Similar functions with different signatures
3. **Documentation debt:** Missing/outdated docstrings
4. **Magic numbers:** Hardcoded values without constants
5. **Copy-paste errors:** Similar code with subtle differences

**Review Checklist:**
- [ ] Check each module's `__init__.py` for API consistency
- [ ] Review base classes for abstraction opportunities
- [ ] Look for similar function names across modules
- [ ] Check for inconsistent parameter naming
- [ ] Identify common validation patterns

### Phase 3: Categorization & Scoring (2-3 hours)

For each issue found:

1. **Category:** DEPRECATION | DRY | COMPLEXITY | UNUSED | TEST-DEBT
2. **Effort:** Low | Medium | High | Very High (use time estimates above)
3. **Impact:** Low | Medium | High | Critical (use criteria above)
4. **Quadrant:** Quick Win | Big Bet | Fill-in | Time Sink
5. **Dependencies:** List any blockers or related issues
6. **Notes:** Any context, risks, or considerations

**Use this scoring formula for tie-breaking:**
```
Priority Score = (Impact × 3 + Knowledge × 2) - (Effort × 2)

Where:
- Impact: Critical=4, High=3, Medium=2, Low=1
- Knowledge: High familiarity=3, Medium=2, Low=1
- Effort: Very High=4, High=3, Medium=2, Low=1

Higher score = higher priority
```

### Phase 4: Tracking & Reporting (1-2 hours)

Create a tracking spreadsheet or GitHub issues:

**Recommended Format (CSV/Spreadsheet):**
```csv
ID,Category,Description,File,Lines,Effort,Impact,Quadrant,Score,Status,Notes
1,DRY,Duplicate validation logic,utils/label.py,45-60,Low,High,Quick Win,9,Open,Extract to shared function
2,COMPLEXITY,Nested loops in plot,plots/heatmap.py,120-180,Medium,Medium,Fill-in,4,Open,Consider vectorization
3,DEPRECATION,Old pandas .ix usage,analysis/customer.py,89,Low,Critical,Quick Win,11,Open,Replace with .loc
```

**GitHub Issue Template:**
```markdown
## [CATEGORY] Brief Description

**Location:** `file/path.py:line_number`
**Effort:** Low | Medium | High | Very High
**Impact:** Low | Medium | High | Critical
**Quadrant:** Quick Win | Big Bet | Fill-in | Time Sink

### Current State
[Describe the problematic code or pattern]

### Proposed Improvement
[Describe the refactoring or fix]

### Benefits
- List specific improvements
- Quantify if possible (e.g., "reduces duplication by 150 lines")

### Risks & Considerations
- Any potential breaking changes
- Dependencies on other issues

### Acceptance Criteria
- [ ] Code changes implemented
- [ ] Tests updated/added
- [ ] Documentation updated
- [ ] No regression in coverage
```

---

## 4. Recommended Tooling Setup

Install these tools for ongoing monitoring:

```bash
# One-time setup
pip install pylint radon vulture

# Add to pre-commit config (.pre-commit-config.yaml)
repos:
  - repo: local
    hooks:
      - id: complexity-check
        name: Check cyclomatic complexity
        entry: radon cc -n C pyretailscience/
        language: system
        types: [python]
        pass_filenames: false

      - id: duplicate-check
        name: Check for code duplication
        entry: pylint --disable=all --enable=duplicate-code --fail-under=9.5 pyretailscience/
        language: system
        types: [python]
        pass_filenames: false
```

**CI/CD Integration:**
```yaml
# .github/workflows/code-quality.yml
name: Code Quality

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install pylint radon vulture
          pip install -e .
      - name: Check duplication
        run: pylint --disable=all --enable=duplicate-code pyretailscience/
      - name: Check complexity
        run: radon cc pyretailscience/ -a -nb -s
      - name: Find dead code
        run: vulture pyretailscience/ --min-confidence 80
```

---

## 5. Quick Start Guide

**For Your First Assessment:**

1. **Run the automated script** (Phase 1):
   ```bash
   chmod +x scripts/assess_technical_debt.sh
   ./scripts/assess_technical_debt.sh
   ```

2. **Review top 3 categories manually** (Phase 2):
   - Code duplication (search for duplicates in similar modules)
   - Complexity (check functions >50 lines)
   - Dead code (check vulture output)

3. **Create prioritized list** (Phase 3):
   - List all Quick Wins (aim for 5-10 items)
   - Identify 2-3 Big Bets for sprint planning
   - Note Fill-ins for idle time

4. **Start with one Quick Win** to validate process

5. **Track progress** in GitHub issues or project board

---

## 6. Success Metrics

Track these KPIs over time:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Code Duplication | <5% | `pylint --duplicate-code` |
| Avg Complexity | ≤5 | `radon cc -a` |
| Dead Code Instances | 0 | `vulture --min-confidence 80` |
| Test Coverage | >85% | `pytest --cov` |
| Ruff Violations | 0 | `ruff check .` |
| High Complexity Functions | 0 | `radon cc -nb -nc` (C/D/F grades) |

**Monthly Review:**
- Generate debt report
- Track trend: improving or degrading?
- Celebrate wins: lines removed, complexity reduced
- Plan next month's Big Bets

---

## 7. Common Patterns in Retail Science Code

**Look for these specific patterns in PyRetailScience:**

1. **Repeated DataFrame operations:**
   ```python
   # Anti-pattern: Duplicated groupby/agg logic
   df.groupby('customer_id').agg({'sales': 'sum'})
   # Appears in multiple modules
   ```

2. **Similar segmentation logic:**
   ```python
   # Check rfm.py, hml.py, threshold.py for shared patterns
   # Opportunity for base class abstraction
   ```

3. **Plot configuration duplication:**
   ```python
   # plots/*.py likely share styling setup
   # Extract to plots/styles or use context managers
   ```

4. **Validation patterns:**
   ```python
   # Repeated checks across analysis modules:
   if df is None: raise ValueError(...)
   if not isinstance(df, pd.DataFrame): ...
   # Extract to utils/validation.py
   ```

5. **Date manipulation:**
   ```python
   # Check utils/date.py vs inline date logic elsewhere
   # Consolidate to single module
   ```

---

## 8. Example Assessment Output

```
TECHNICAL DEBT ASSESSMENT - PyRetailScience
Date: 2025-11-17
Codebase: 44 files, ~10,840 lines

SUMMARY:
========
Total Issues: 23
├─ Quick Wins: 8 (35%)
├─ Big Bets: 5 (22%)
├─ Fill-ins: 7 (30%)
└─ Time Sinks: 3 (13%)

BY CATEGORY:
===========
DRY Violations: 9 (39%)
Complexity: 6 (26%)
Dead Code: 4 (17%)
Deprecation: 2 (9%)
Test Debt: 2 (9%)

TOP PRIORITY (Quick Wins):
========================
1. [DRY] Extract repeated DataFrame validation (3 files, 45 lines)
   Effort: Low | Impact: High | Score: 9

2. [DEPRECATION] Replace pd.append with pd.concat (2 files)
   Effort: Low | Impact: Critical | Score: 11

3. [COMPLEXITY] Simplify nested loops in heatmap.py (45 lines → 15 lines)
   Effort: Low | Impact: High | Score: 9

... (continue for all Quick Wins)

RECOMMENDED NEXT STEPS:
======================
Week 1: Address Quick Wins #1-3
Week 2: Address Quick Wins #4-8 + plan Big Bet #1
Week 3: Execute Big Bet #1
Week 4: Review progress, update metrics
```

---

## Conclusion

This framework provides:
✅ Systematic discovery process
✅ Clear categorization and scoring
✅ Effort vs impact prioritization
✅ Actionable tooling and automation
✅ Ongoing monitoring approach
✅ Retail-science-specific patterns

**Start small, measure impact, iterate.** Even addressing 2-3 Quick Wins per week will compound into significant improvements over time.

**Questions or need help?** Reference the specific sections above or run the automated assessment to get started immediately.
