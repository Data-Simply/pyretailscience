# PyRetailScience Release Process

This document explains the deployment strategy, branching model, and release process for PyRetailScience. It is designed to support controlled feature releases, hotfixes, and maintain a single stable version.

## Table of Contents

- [Branching Strategy](#branching-strategy)
- [Release Workflow](#release-workflow)
- [Hotfix Workflow](#hotfix-workflow)
- [Changelog Management](#changelog-management)
- [Social Content Generation](#social-content-generation)
- [Quick Reference](#quick-reference)

---

## Branching Strategy

PyRetailScience follows a **GitHub Flow + Release Branches** model, which provides:
- ✅ Controlled feature deployment
- ✅ Clear hotfix path
- ✅ Low maintenance overhead
- ✅ Manual gatekeeping for releases

### Branch Types

```
main (development branch - always "release-ready")
├── release/v0.40.x (stable release branch for v0.40)
├── release/v0.41.x (stable release branch for v0.41)
├── feature/* (feature development)
└── hotfix/* (critical bug fixes)
```

### Branch Descriptions

| Branch Type | Purpose | Base Branch | Merge Target | Lifetime |
|------------|---------|-------------|--------------|----------|
| `main` | Development integration, release-ready code | N/A | N/A | Permanent |
| `release/v*.*.x` | Stable release maintenance | `main` at release time | N/A | Permanent |
| `feature/*` | Feature development | `main` | `main` | Short-lived |
| `hotfix/*` | Critical bug fixes | Latest `release/*` | `release/*` → `main` | Short-lived |

### Key Principles

1. **`main` is always release-ready** but not "released" until you create a release branch
2. **Features merge to `main`** when complete, but this doesn't trigger deployment
3. **Releases happen from release branches** created from `main` at chosen milestones
4. **Only maintain the latest release branch** (we don't backport to older versions)
5. **Hotfixes go to release branch first**, then merge forward to `main`

---

## Release Workflow

### When to Release

Create a release when:
- You've accumulated enough features for a version bump
- A milestone is complete
- Manual decision by maintainers (we use manual gatekeeping)

### Standard Release Process

#### Step 1: Prepare Release Branch

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create release branch from main
# Use semantic versioning: major.minor.x
git checkout -b release/v0.41.x
git push -u origin release/v0.41.x
```

#### Step 2: Generate and Review Changelog

```bash
# Generate changelog draft (requires git-cliff installed)
git-cliff --tag v0.41.0 --unreleased --output CHANGELOG_DRAFT.md

# Review and edit the changelog
# - Add context to terse commit messages
# - Group related changes
# - Add migration notes if needed
# - Highlight breaking changes

# Once satisfied, update the docs changelog
git-cliff --tag v0.41.0 --unreleased --output docs/changelog.md
```

**Important:** The changelog is generated automatically but should be reviewed and edited before committing. See [Changelog Management](#changelog-management) for details.

#### Step 3: Run Release Workflow

```bash
# Commit the changelog
git add docs/changelog.md
git commit -m "docs: add changelog for v0.41.0"
git push origin release/v0.41.x

# Trigger the release workflow via GitHub UI:
# 1. Go to Actions → Release
# 2. Click "Run workflow"
# 3. Select branch: release/v0.41.x
# 4. Choose version bump: minor (0.40.0 → 0.41.0)
# 5. Click "Run workflow"
```

The automated workflow will:
1. Run pre-commit checks (linting, formatting, tests)
2. Run BigQuery integration tests
3. Run PySpark integration tests
4. Bump version in `pyproject.toml`
5. Build the package
6. Create git tag (e.g., `v0.41.0`)
7. Create GitHub release with notes
8. Publish to PyPI

#### Step 4: Verify Release

```bash
# Check PyPI
open https://pypi.org/project/pyretailscience/

# Test installation in clean environment
uv venv test-env
source test-env/bin/activate
pip install pyretailscience==0.41.0

# Verify it works
python -c "import pyretailscience; print(pyretailscience.__version__)"
```

#### Step 5: Merge Changelog Back to Main

```bash
# Ensure main has the updated changelog
git checkout main
git merge release/v0.41.x --no-ff -m "docs: merge v0.41.0 changelog to main"
git push origin main
```

#### Step 6: Generate Social Content (Optional)

See [Social Content Generation](#social-content-generation) for creating announcements.

---

## Hotfix Workflow

### When to Use Hotfixes

Use the hotfix workflow for:
- Critical bugs in the latest release
- Security vulnerabilities
- Data corruption issues
- Broken functionality that blocks users

**Do NOT use hotfixes for:**
- Feature requests (wait for next release)
- Minor bugs that don't block users
- Documentation updates (can go through main)

### Hotfix Process

#### Step 1: Create Hotfix Branch

```bash
# Start from the latest release branch
git checkout release/v0.41.x
git pull origin release/v0.41.x

# Create hotfix branch
git checkout -b hotfix/fix-critical-calculation-bug
```

#### Step 2: Implement and Test Fix

```bash
# Make your changes
# ... edit files ...

# Test thoroughly
uv run pytest
uv run pytest --cov=pyretailscience

# Commit using conventional commits
git add .
git commit -m "fix: correct revenue calculation in segmentation module

Fixes issue where revenue was double-counted when transactions
spanned multiple segments. Added test case to prevent regression.

Fixes #123"
```

#### Step 3: Create PR to Release Branch

```bash
# Push hotfix branch
git push -u origin hotfix/fix-critical-calculation-bug

# Create PR targeting release/v0.41.x (not main!)
gh pr create --base release/v0.41.x --title "Hotfix: Correct revenue calculation" --body "..."
```

#### Step 4: Merge and Release

```bash
# After PR approval, merge to release branch
gh pr merge --squash

# Switch to release branch
git checkout release/v0.41.x
git pull origin release/v0.41.x

# Update changelog
git-cliff --tag v0.41.1 --unreleased --output docs/changelog.md
git add docs/changelog.md
git commit -m "docs: add changelog for v0.41.1"
git push origin release/v0.41.x

# Trigger release workflow with PATCH bump
# Actions → Release → Run workflow → select release/v0.41.x → patch
```

#### Step 5: Merge Forward to Main

```bash
# Ensure main has the fix
git checkout main
git pull origin main

# Merge release branch to main
git merge release/v0.41.x --no-ff -m "fix: merge hotfix v0.41.1 to main"

# Resolve conflicts if any, then push
git push origin main
```

---

## Changelog Management

### Overview

We use **git-cliff** to automatically generate changelogs from conventional commits. The changelog is:
- **Auto-generated** from git history
- **Manually reviewable** before committing
- **Integrated** into MkDocs documentation
- **Searchable** via docs site search

### Setup git-cliff (One-Time)

```bash
# Install git-cliff (macOS)
brew install git-cliff

# Install git-cliff (Linux)
cargo install git-cliff

# Install git-cliff (via binary)
# Download from: https://github.com/orhun/git-cliff/releases
```

### Changelog Configuration

The changelog format is defined in `.cliff.toml`:

```toml
[changelog]
header = "# Changelog\n\nAll notable changes to PyRetailScience.\n"
body = """
## [{{ version }}] - {{ timestamp | date(format="%Y-%m-%d") }}

{% for group, commits in commits | group_by(attribute="group") %}
### {{ group | upper_first }}
{% for commit in commits %}
  - {{ commit.message | split(pat="\n") | first | trim }}\
    {% if commit.breaking %} **BREAKING**{% endif %}
{% endfor %}
{% endfor %}
"""

[git]
conventional_commits = true
filter_unconventional = true
commit_parsers = [
  { message = "^feat", group = "Features" },
  { message = "^fix", group = "Bug Fixes" },
  { message = "^docs", group = "Documentation" },
  { message = "^perf", group = "Performance" },
  { message = "^refactor", group = "Refactoring" },
  { message = "^test", group = "Testing" },
  { message = "^chore", skip = true },
  { message = "^style", skip = true },
  { message = "^build", skip = true },
]
```

### Generating and Editing Changelogs

#### Step 1: Generate Draft

```bash
# Generate changelog for upcoming version (not committed yet)
git-cliff --tag v0.41.0 --unreleased --output CHANGELOG_DRAFT.md

# Or for a specific range
git-cliff v0.40.0..HEAD --tag v0.41.0 --output CHANGELOG_DRAFT.md
```

This creates `CHANGELOG_DRAFT.md` in your working directory.

#### Step 2: Review and Edit

Open `CHANGELOG_DRAFT.md` in your editor and improve it:

**Before (raw git-cliff output):**
```markdown
## [0.41.0] - 2025-11-13

### Features
  - add segment filtering
  - support custom metrics
  - new bar plot options

### Bug Fixes
  - fix type hint
  - update docs
```

**After (edited for clarity):**
```markdown
## [0.41.0] - 2025-11-13

### Features
  - **Segmentation**: Add ability to filter segments by custom criteria ([#145](https://github.com/Data-Simply/pyretailscience/pull/145))
  - **Metrics**: Support custom metric calculations in revenue tree analysis ([#148](https://github.com/Data-Simply/pyretailscience/pull/148))
  - **Plotting**: New `legend_position` and `color_scheme` options for bar plots ([#151](https://github.com/Data-Simply/pyretailscience/pull/151))

### Bug Fixes
  - **Type Safety**: Fix type hints for `SegmentationBase.filter()` method ([#149](https://github.com/Data-Simply/pyretailscience/pull/149))
  - **Documentation**: Update installation guide with Python 3.12 support notes ([#150](https://github.com/Data-Simply/pyretailscience/pull/150))

### Migration Notes
  - The `legend` parameter in `bar_plot()` is now deprecated in favor of `legend_position`. The old parameter still works but will be removed in v0.42.0.
```

**Editing Tips:**
- Add context to terse commit messages
- Link to PRs or issues with `[#123](...)`
- Group related changes under bold subsections
- Add **Migration Notes** for breaking changes
- Include code examples for new features
- Remove internal/uninteresting changes

#### Step 3: Commit the Final Changelog

```bash
# Once satisfied with edits, copy to docs
cp CHANGELOG_DRAFT.md docs/changelog.md

# Commit it
git add docs/changelog.md
git commit -m "docs: add changelog for v0.41.0"
git push

# Clean up draft
rm CHANGELOG_DRAFT.md
```

### Changelog in MkDocs

The changelog will be available at: `https://pyretailscience.datasimply.co/changelog/`

Update `mkdocs.yml` to include it in navigation:

```yaml
nav:
  - Home: index.md
  - Changelog: changelog.md
  - Getting Started:
      - Installation: getting_started/installation.md
      - Options: getting_started/options_guide.md
  # ... rest of nav
```

---

## Social Content Generation

### Overview

Turn your changelog into engaging content for:
- Twitter/X threads
- LinkedIn posts
- GitHub Discussions
- Blog posts
- Release announcement emails

### Strategy 1: Automated Social Post Generation

**Tool:** LLM-based (Claude, GPT) with structured prompt

Create a script to generate social content from changelog:

```bash
# generate_social_content.sh
#!/bin/bash

VERSION=$1
CHANGELOG_FILE="docs/changelog.md"

# Extract the latest version section
LATEST_SECTION=$(awk "/## \[$VERSION\]/,/## \[/" "$CHANGELOG_FILE" | head -n -1)

# Generate social posts using LLM (requires API key)
echo "$LATEST_SECTION" | claude-cli --prompt "
Convert this changelog into:
1. A Twitter/X thread (3-5 tweets, engaging, use emojis sparingly)
2. A LinkedIn post (professional, highlight business value)
3. A short GitHub Discussion post

Changelog:
" > social_content_${VERSION}.md

echo "Social content saved to social_content_${VERSION}.md"
```

### Strategy 2: Manual Templates

Create templates for different platforms:

#### Twitter/X Thread Template

```markdown
🚀 PyRetailScience v0.41.0 is here!

Key highlights:
🔹 [Feature 1 - business value]
🔹 [Feature 2 - business value]
🔹 [Fix - impact]

[1/4]

---

Deep dive: [Feature 1]

[Short description with code example or visual]

This unlocks [business value for retail analysts]

[2/4]

---

[Continue for major features]

---

Full changelog: https://pyretailscience.datasimply.co/changelog/
Install: pip install pyretailscience==0.41.0

[4/4]
```

#### LinkedIn Post Template

```markdown
We're excited to announce PyRetailScience v0.41.0! 🎉

This release brings powerful new capabilities for retail analytics teams:

**New Segmentation Filtering**
Analysts can now apply custom filters to customer segments, enabling more granular cohort analysis. This is especially useful when analyzing regional differences or product category trends.

**Custom Metrics in Revenue Trees**
Define your own KPIs and visualize them in the revenue tree hierarchy—perfect for tracking non-standard metrics like customer lifetime value or loyalty program engagement.

**Enhanced Bar Plotting**
More control over visualizations with new legend positioning and color scheme options. Your reports just got more beautiful.

Plus numerous bug fixes and performance improvements.

Read the full changelog: [link]
Get started: pip install pyretailscience==0.41.0

#RetailAnalytics #DataScience #Python #OpenSource
```

#### GitHub Discussion Post Template

```markdown
# PyRetailScience v0.41.0 Released 🎉

We're happy to announce the release of PyRetailScience v0.41.0!

## What's New

### Features
- **Segment Filtering**: Apply custom filters to segments for more granular analysis
- **Custom Metrics**: Define your own metrics in revenue tree analysis
- **Plot Enhancements**: New options for bar plot customization

### Bug Fixes
- Fixed type hints in segmentation module
- Updated documentation for Python 3.12 support

## Installation

```bash
pip install --upgrade pyretailscience
```

## Full Changelog

See the complete changelog here: [link]

## Feedback

Try it out and let us know what you think! We'd love to hear:
- What features are you most excited about?
- Any issues or suggestions?

Drop a comment below or open an issue on GitHub.
```

### Strategy 3: Semi-Automated Workflow

**Step 1: Generate Draft Posts**

```bash
# After committing changelog, generate social content
./scripts/generate_social_content.sh v0.41.0
```

**Step 2: Review and Edit**

Edit `social_content_v0.41.0.md` to:
- Add personality and voice
- Include relevant hashtags
- Add visuals (screenshots, code examples)
- Tailor for each platform's audience

**Step 3: Schedule Posts**

- Use Buffer, Hootsuite, or similar for scheduling
- Post GitHub Discussion immediately after release
- Schedule Twitter thread for peak engagement time
- Share LinkedIn post 1-2 days later for secondary wave

### Visual Content Ideas

Enhance social posts with:

1. **Feature Screenshots**: GIFs showing new features in action
2. **Code Examples**: Syntax-highlighted code snippets
3. **Before/After**: Show improvements visually
4. **Metrics**: "This release includes X features, Y fixes"
5. **Contributor Shoutouts**: Thank contributors with @mentions

### Content Calendar

| Time | Platform | Content Type |
|------|----------|--------------|
| Release Day | GitHub Discussion | Full announcement |
| Release Day | Twitter/X | Thread (3-5 tweets) |
| Release Day + 1 | LinkedIn | Professional post |
| Release Day + 2 | Blog (if exists) | Deep dive article |
| Release Day + 7 | Twitter/X | Tutorial thread on best feature |

---

## Quick Reference

### Common Commands

```bash
# Create release branch
git checkout -b release/v0.41.x main
git push -u origin release/v0.41.x

# Generate changelog draft
git-cliff --tag v0.41.0 --unreleased --output CHANGELOG_DRAFT.md

# Edit, then commit changelog
git add docs/changelog.md
git commit -m "docs: add changelog for v0.41.0"

# Trigger release (via GitHub Actions UI)
# Actions → Release → Run workflow → select branch → choose bump type

# Verify release
pip install pyretailscience==0.41.0

# Merge changelog to main
git checkout main
git merge release/v0.41.x --no-ff
```

### Release Checklist

- [ ] Create release branch from `main`
- [ ] Generate changelog draft with git-cliff
- [ ] Review and edit changelog for clarity
- [ ] Commit changelog to release branch
- [ ] Trigger release workflow via GitHub Actions
- [ ] Verify release on PyPI
- [ ] Test installation in clean environment
- [ ] Merge changelog back to `main`
- [ ] Generate social content
- [ ] Post release announcements
- [ ] Update any dependent projects (if applicable)

### Hotfix Checklist

- [ ] Create hotfix branch from latest `release/*`
- [ ] Implement fix with tests
- [ ] Create PR to release branch (not main!)
- [ ] Merge PR after approval
- [ ] Generate changelog for patch version
- [ ] Trigger release workflow with PATCH bump
- [ ] Verify hotfix on PyPI
- [ ] Merge release branch forward to `main`
- [ ] Notify users of hotfix (GitHub Discussion, Twitter)

---

## Getting Help

If you have questions about the release process:

1. Check this document first
2. Ask in [GitHub Discussions](https://github.com/Data-Simply/pyretailscience/discussions)
3. Contact a maintainer
4. Open an issue with the `question` label

## Related Documentation

- [CONTRIBUTING.md](CONTRIBUTING.md) - General contribution guidelines
- [CLAUDE.md](CLAUDE.md) - Development and code style guidelines
- [README.md](README.md) - Project overview and quick start

---

**Last Updated:** 2025-11-13
**Maintained By:** PyRetailScience Maintainers
