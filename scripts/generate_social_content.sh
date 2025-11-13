#!/bin/bash
# Generate social media content from changelog
# Usage: ./scripts/generate_social_content.sh v0.41.0

set -e

VERSION=$1
CHANGELOG_FILE="docs/changelog.md"
OUTPUT_DIR="social_content"

# Validate version argument
if [ -z "$VERSION" ]; then
    echo "Error: Version argument required"
    echo "Usage: ./scripts/generate_social_content.sh v0.41.0"
    exit 1
fi

# Strip 'v' prefix if present for consistent handling
VERSION_NUM="${VERSION#v}"

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/social_${VERSION_NUM}.md"

# Check if changelog exists
if [ ! -f "$CHANGELOG_FILE" ]; then
    echo "Error: Changelog file not found at $CHANGELOG_FILE"
    exit 1
fi

# Extract the version section from changelog
echo "Extracting changelog section for version $VERSION_NUM..."

# Use awk to extract the section for this version
CHANGELOG_SECTION=$(awk "/## \[${VERSION_NUM}\]/,/## \[/" "$CHANGELOG_FILE" | sed '$d')

if [ -z "$CHANGELOG_SECTION" ]; then
    echo "Error: Could not find version $VERSION_NUM in changelog"
    exit 1
fi

# Count features and fixes for summary
FEATURE_COUNT=$(echo "$CHANGELOG_SECTION" | grep -c "^- " | grep -v "^0$" || echo "0")

echo "Found $FEATURE_COUNT changes"

# Generate the social content template file
cat > "$OUTPUT_FILE" << 'EOF'
# Social Content for PyRetailScience {VERSION}

**Generated:** {DATE}

---

## Twitter/X Thread

### Tweet 1 (Main Announcement)
```
🚀 PyRetailScience {VERSION} is here!

Key highlights:
🔹 [FEATURE_1_SUMMARY]
🔹 [FEATURE_2_SUMMARY]
🔹 [FIX_SUMMARY]

Thread 🧵 👇

[1/4]
```

### Tweet 2 (Deep Dive - Feature 1)
```
Deep dive: [FEATURE_1_TITLE]

[EXPLAIN_FEATURE_1_VALUE - 2-3 sentences]

[CODE_EXAMPLE_OR_SCREENSHOT_DESCRIPTION]

[2/4]
```

### Tweet 3 (Deep Dive - Feature 2)
```
Also new: [FEATURE_2_TITLE]

[EXPLAIN_FEATURE_2_VALUE - 2-3 sentences]

Perfect for: [USE_CASE]

[3/4]
```

### Tweet 4 (Call to Action)
```
Full changelog: https://pyretailscience.datasimply.co/changelog/

Install: pip install pyretailscience=={VERSION}

Try it out and let us know what you think! 💬

[4/4]
```

---

## LinkedIn Post

```markdown
We're excited to announce PyRetailScience {VERSION}! 🎉

This release brings powerful new capabilities for retail analytics teams:

**[FEATURE_1_TITLE]**
[2-3 sentences explaining business value and use case]

**[FEATURE_2_TITLE]**
[2-3 sentences explaining business value and use case]

**Bug Fixes & Improvements**
[Summary of important fixes]

Plus numerous performance improvements and documentation updates.

📚 Read the full changelog: https://pyretailscience.datasimply.co/changelog/
⚡ Get started: pip install pyretailscience=={VERSION}

#RetailAnalytics #DataScience #Python #OpenSource #RetailTech
```

---

## GitHub Discussion Post

```markdown
# PyRetailScience {VERSION} Released 🎉

We're happy to announce the release of PyRetailScience {VERSION}!

## What's New

### Features
- **[FEATURE_1]**: [Description]
- **[FEATURE_2]**: [Description]
- **[FEATURE_3]**: [Description]

### Bug Fixes
- [FIX_1]
- [FIX_2]

### Documentation
- [DOC_UPDATE_1]
- [DOC_UPDATE_2]

## Installation

```bash
pip install --upgrade pyretailscience
# or
pip install pyretailscience=={VERSION}
```

## Full Changelog

See the complete changelog here: https://pyretailscience.datasimply.co/changelog/

## Feedback

Try it out and let us know what you think! We'd love to hear:
- What features are you most excited about?
- Any issues or suggestions?
- How are you using PyRetailScience in your work?

Drop a comment below or open an issue on GitHub.

## Contributors

Thank you to everyone who contributed to this release! 🙏
[LIST_CONTRIBUTORS]
```

---

## Blog Post Outline (Optional)

**Title:** What's New in PyRetailScience {VERSION}

**Introduction:**
- Brief overview of the release
- Key themes (e.g., "This release focuses on visualization and performance")

**Section 1: [Major Feature 1]**
- What it does
- Why it matters
- Code example
- Screenshot/visualization

**Section 2: [Major Feature 2]**
- What it does
- Why it matters
- Code example
- Screenshot/visualization

**Section 3: Other Improvements**
- Bullet list of smaller features
- Bug fixes
- Performance improvements

**Conclusion:**
- How to upgrade
- Where to learn more
- Call for feedback

---

## Release Email (if mailing list exists)

**Subject:** PyRetailScience {VERSION} Released - [Key Feature]

**Body:**
```
Hi there,

We're excited to announce PyRetailScience {VERSION} is now available!

What's New:

🎯 [Feature 1 with value prop]
📊 [Feature 2 with value prop]
🐛 [Important fix with impact]

To upgrade:
pip install --upgrade pyretailscience

Full changelog:
https://pyretailscience.datasimply.co/changelog/

Questions or feedback?
Reply to this email or join the discussion:
https://github.com/Data-Simply/pyretailscience/discussions

Happy analyzing!
The PyRetailScience Team
```

---

## Changelog Section (for reference)

EOF

# Replace placeholders with actual version and date
sed -i "s/{VERSION}/$VERSION_NUM/g" "$OUTPUT_FILE"
sed -i "s/{DATE}/$(date +%Y-%m-%d)/g" "$OUTPUT_FILE"

# Append the actual changelog section
echo "$CHANGELOG_SECTION" >> "$OUTPUT_FILE"

echo ""
echo "✅ Social content template generated: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "1. Review and customize the content in $OUTPUT_FILE"
echo "2. Add specific details, code examples, and business value"
echo "3. Create visuals (screenshots, GIFs) if needed"
echo "4. Copy content to your social media scheduling tool"
echo ""
echo "Tips:"
echo "- Focus on business value, not just technical details"
echo "- Use concrete examples and use cases"
echo "- Tag relevant people/organizations"
echo "- Schedule posts for peak engagement times"
echo ""
