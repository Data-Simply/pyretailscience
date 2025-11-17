#!/bin/bash
# Technical Debt Assessment Script for PyRetailScience
# Runs automated tools to identify deprecation, duplication, complexity, and dead code

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_FILE="technical_debt_report.txt"
HTML_OUTPUT="technical_debt_report.html"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PyRetailScience Technical Debt Assessment                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated: $(date)"
echo "Repository: $(pwd)"
echo ""

# Check if tools are installed
echo "🔍 Checking required tools..."
MISSING_TOOLS=""

if ! command -v pylint &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS pylint"
fi

if ! command -v radon &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS radon"
fi

if ! command -v vulture &> /dev/null; then
    MISSING_TOOLS="$MISSING_TOOLS vulture"
fi

if [ -n "$MISSING_TOOLS" ]; then
    echo "⚠️  Missing tools:$MISSING_TOOLS"
    echo "Installing missing tools..."
    pip install $MISSING_TOOLS
    echo "✅ Tools installed"
fi

# Initialize report
cat > "$OUTPUT_FILE" << 'EOF'
================================================================================
                    TECHNICAL DEBT ASSESSMENT REPORT
                         PyRetailScience
================================================================================

EOF

echo "Generated: $(date)" >> "$OUTPUT_FILE"
echo "Repository: $(pwd)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 1. Code Statistics
echo "📊 Gathering code statistics..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "1. CODE STATISTICS" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

FILE_COUNT=$(find pyretailscience/ -name "*.py" -type f | wc -l)
LINE_COUNT=$(find pyretailscience/ -name "*.py" -type f -exec wc -l {} + | tail -1 | awk '{print $1}')
TEST_COUNT=$(find tests/ -name "test_*.py" -type f | wc -l)

echo "Python files: $FILE_COUNT" >> "$OUTPUT_FILE"
echo "Total lines of code: $LINE_COUNT" >> "$OUTPUT_FILE"
echo "Test files: $TEST_COUNT" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 2. Code Duplication
echo "🔄 Analyzing code duplication..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "2. CODE DUPLICATION ANALYSIS" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

pylint --disable=all --enable=duplicate-code pyretailscience/ 2>&1 | tee -a "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# 3. Complexity Analysis
echo "📈 Analyzing code complexity..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "3. COMPLEXITY ANALYSIS" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "--- Average Complexity ---" >> "$OUTPUT_FILE"
radon cc pyretailscience/ -a -s >> "$OUTPUT_FILE" 2>&1 || true
echo "" >> "$OUTPUT_FILE"

echo "--- High Complexity Functions (Grade C or worse) ---" >> "$OUTPUT_FILE"
radon cc pyretailscience/ -nc -nb >> "$OUTPUT_FILE" 2>&1 || true
echo "" >> "$OUTPUT_FILE"

# 4. Maintainability Index
echo "🔧 Calculating maintainability index..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "4. MAINTAINABILITY INDEX" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

radon mi pyretailscience/ -s >> "$OUTPUT_FILE" 2>&1 || true
echo "" >> "$OUTPUT_FILE"

# 5. Dead Code Detection
echo "💀 Detecting dead code..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "5. DEAD CODE DETECTION" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

vulture pyretailscience/ --min-confidence 80 >> "$OUTPUT_FILE" 2>&1 || true
echo "" >> "$OUTPUT_FILE"

# 6. Linting Issues
echo "🔍 Running linting checks..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "6. LINTING ISSUES (RUFF)" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

uv run ruff check . --statistics >> "$OUTPUT_FILE" 2>&1 || true
echo "" >> "$OUTPUT_FILE"

# 7. Test Coverage
echo "🧪 Analyzing test coverage..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "7. TEST COVERAGE ANALYSIS" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

uv run pytest --cov=pyretailscience --cov-report=term-missing --cov-report=html:htmlcov 2>&1 | tee -a "$OUTPUT_FILE" || true
echo "" >> "$OUTPUT_FILE"

# 8. Potential Deprecations
echo "⚠️  Searching for deprecation markers..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "8. POTENTIAL DEPRECATIONS & TECHNICAL DEBT MARKERS" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "--- TODO/FIXME/XXX/HACK markers ---" >> "$OUTPUT_FILE"
rg "TODO|FIXME|XXX|HACK" pyretailscience/ -n --type py >> "$OUTPUT_FILE" 2>&1 || echo "None found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "--- Deprecation warnings ---" >> "$OUTPUT_FILE"
rg "DeprecationWarning|deprecated|@deprecated" pyretailscience/ -n --type py >> "$OUTPUT_FILE" 2>&1 || echo "None found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "--- Potentially deprecated pandas patterns ---" >> "$OUTPUT_FILE"
rg "\.ix\[|\.append\(|inplace=True" pyretailscience/ -n --type py >> "$OUTPUT_FILE" 2>&1 || echo "None found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# 9. Long Functions
echo "📏 Finding long functions (>50 lines)..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "9. LONG FUNCTIONS (>50 LINES)" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

find pyretailscience/ -name "*.py" -exec awk '
/^def |^    def / {
    if (start && name && (NR - start > 50)) {
        print FILENAME ":" start ":" name " (" (NR-start) " lines)"
    }
    start = NR
    name = $0
}
END {
    if (start && name && (NR - start > 50)) {
        print FILENAME ":" start ":" name " (" (NR-start) " lines)"
    }
}' {} + >> "$OUTPUT_FILE" 2>&1 || echo "None found" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"

# 10. Deep Nesting
echo "🪺 Finding deeply nested code (>4 levels)..."
echo "================================================================================" >> "$OUTPUT_FILE"
echo "10. DEEP NESTING (>4 INDENTATION LEVELS)" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

rg "^\s{16,}" pyretailscience/ --type py -n >> "$OUTPUT_FILE" 2>&1 || echo "None found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Summary
echo "================================================================================" >> "$OUTPUT_FILE"
echo "SUMMARY" >> "$OUTPUT_FILE"
echo "================================================================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF
Review complete!

KEY ACTIONS:
1. Review code duplication section for extract-to-function opportunities
2. Address high complexity functions (Grade C or worse)
3. Investigate dead code findings (min confidence: 80%)
4. Fix any ruff violations
5. Improve test coverage in areas below 80%
6. Refactor long functions (>50 lines) and deeply nested code
7. Address deprecated patterns and technical debt markers

NEXT STEPS:
- Review this report in detail
- Categorize findings using the framework (TECHNICAL_DEBT_FRAMEWORK.md)
- Create GitHub issues for Quick Wins
- Prioritize using Effort vs Impact matrix
- Start with 2-3 Quick Wins this week

Coverage report: htmlcov/index.html
Full report: $(pwd)/$OUTPUT_FILE
EOF

echo "" >> "$OUTPUT_FILE"
echo "Report generation complete: $(date)" >> "$OUTPUT_FILE"

# Display summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Assessment Complete!                                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📄 Full report: $OUTPUT_FILE"
echo "📊 Coverage report: htmlcov/index.html"
echo ""
echo "Quick Stats:"
echo "  • Files analyzed: $FILE_COUNT"
echo "  • Lines of code: $LINE_COUNT"
echo "  • Test files: $TEST_COUNT"
echo ""
echo "Next: Review $OUTPUT_FILE and categorize findings using the framework"
echo "      (see TECHNICAL_DEBT_FRAMEWORK.md for guidance)"
echo ""
