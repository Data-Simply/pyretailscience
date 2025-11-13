# Release Guide for New Developers

Welcome! This guide will help you understand how releases work in PyRetailScience. If you're contributing code, understanding this workflow will help you see how your changes eventually reach users.

## TL;DR - The Big Picture

```
┌─────────────┐
│   feature   │  ← You work here (feature branches)
│   branches  │
└──────┬──────┘
       │ PR & merge
       ▼
┌─────────────┐
│    main     │  ← Code merges here when ready
│ (development)│  ← NOT auto-deployed!
└──────┬──────┘
       │ When ready to release...
       ▼
┌─────────────┐
│  release/   │  ← Release happens from here
│   v0.41.x   │  ← This gets published to PyPI
└─────────────┘
```

**Key Point:** Merging to `main` does NOT deploy to users. Releases are manual and controlled.

---

## The Workflow in 5 Steps

### 1️⃣ Develop Your Feature

```bash
# Start from main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/add-cool-analysis

# Make changes, commit (we use conventional commits!)
git add .
git commit -m "feat: add cool new analysis function"

# Push and create PR
git push -u origin feature/add-cool-analysis
gh pr create --base main
```

**Your role:** Write code, tests, and get PR reviewed.

### 2️⃣ Merge to Main

After approval, your PR gets merged to `main`.

**What happens:**
- ✅ Your code is now in `main`
- ✅ Pre-commit checks run
- ✅ Tests pass
- ❌ NOT deployed to users yet!

**Why?** We want to control WHEN releases happen, not automatically deploy every merge.

### 3️⃣ Release Decision (Maintainers)

Maintainers decide when to release based on:
- Enough features accumulated
- Milestone complete
- Important bug fixes ready

**Not your concern as a contributor** - maintainers handle this!

### 4️⃣ Create Release (Maintainers)

Maintainers create a release branch:

```bash
git checkout -b release/v0.41.x main
git push -u origin release/v0.41.x
```

Then they:
1. Generate changelog
2. Review and edit it
3. Trigger release workflow via GitHub Actions
4. Package gets published to PyPI

### 5️⃣ Your Feature is Live! 🎉

Users can now install it:

```bash
pip install pyretailscience==0.41.0
```

---

## What You Need to Know

### Conventional Commits (Important!)

We use conventional commit messages. This is enforced by pre-commit hooks.

**Format:** `type(scope): description`

**Types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvement

**Examples:**
```bash
git commit -m "feat: add revenue tree analysis function"
git commit -m "fix: correct calculation in RFM segmentation"
git commit -m "docs: update installation guide for Python 3.12"
git commit -m "test: add tests for cross-shop analysis"
```

**Why it matters:** These commit messages auto-generate the changelog!

### Branch Naming

Follow these patterns:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feature/*` | `feature/add-revenue-tree` |
| Bug fix | `fix/*` | `fix/rfm-calculation-error` |
| Documentation | `docs/*` | `docs/update-api-reference` |
| Hotfix | `hotfix/*` | `hotfix/critical-security-fix` |

### Testing Requirements

Before your PR can merge:

1. ✅ All tests must pass: `uv run pytest`
2. ✅ Code coverage maintained: `uv run pytest --cov=pyretailscience`
3. ✅ Linting passes: `uv run ruff check .`
4. ✅ Formatting correct: `uv run ruff format .`
5. ✅ Pre-commit hooks pass

**Pro tip:** Run `uv run pytest` locally before pushing to catch issues early!

---

## Common Scenarios

### Scenario 1: "My feature is in main, when will it be released?"

**Answer:** When maintainers create the next release. Could be days or weeks depending on what else is being bundled.

**Want to know?** Check GitHub Discussions or ask in your PR: "When is this planned for release?"

### Scenario 2: "There's a critical bug in production!"

**What happens:**
1. Maintainers create a `hotfix/*` branch from the latest `release/*` branch
2. Fix is implemented and tested
3. Hotfix is merged to release branch
4. New patch version is released (e.g., v0.41.0 → v0.41.1)
5. Fix is merged forward to `main`

**Timeline:** Hotfixes can be released same day for critical issues.

### Scenario 3: "I want to see what's planned for the next release"

**Check:**
1. Look at merged PRs to `main` since last release
2. Check GitHub Milestones (if used)
3. Ask in GitHub Discussions

### Scenario 4: "Can I help with a release?"

**Yes!** While maintainers do final releases, you can help with:
- Reviewing the auto-generated changelog for clarity
- Writing release notes or blog posts
- Creating social media content
- Testing pre-release versions

Ask maintainers if you want to get involved!

---

## Visualizing the Branches

```
main:     ●─────●─────●─────●─────●─────●─────●
           \     \           \           \
            \     \           \           \
feature-1:   ●─●─●/            \           \
                                \           \
feature-2:                       ●─●─●─●─●─/\
                                             \
                                              \
release/v0.41.x:                               ●───●───●
                                               ^   ^   ^
                                            v0.41.0  │  v0.41.2
                                                  v0.41.1
                                                 (hotfix)
```

**Explanation:**
- `main` accumulates features over time
- Feature branches merge to `main` when ready
- Release branch created at chosen point
- Hotfixes go to release branch, then merge forward to `main`

---

## FAQs

### Q: Why not deploy from `main` directly?

**A:** We want control over:
- What features ship together
- When releases happen
- Ability to hotfix without deploying unfinished features from `main`

### Q: What if my feature needs to wait for next release?

**A:** That's normal! Features often accumulate in `main` and ship in batches. If your feature is time-sensitive, mention it in the PR and maintainers can expedite.

### Q: Can I make a release?

**A:** Releases require maintainer permissions (PyPI credentials, GitHub release permissions). But you can help with changelog editing, testing, and social content!

### Q: What's the release cadence?

**A:** No fixed schedule. Releases happen when:
- Enough features accumulated (typically 5-10 features)
- Important bug fixes ready
- Milestone completed
- Maintainer discretion

Typically every 2-4 weeks, but can vary.

### Q: How do I know if my change is a breaking change?

**Breaking changes:**
- Remove or rename public functions/classes
- Change function signatures (parameters, return types)
- Change default behavior that users rely on

**Not breaking:**
- Add new functions/parameters (with defaults)
- Fix bugs
- Internal refactoring
- Documentation updates

**If breaking:** Add `BREAKING CHANGE:` in commit message body:
```bash
git commit -m "feat: change revenue_tree API

BREAKING CHANGE: revenue_tree() now requires period parameter.
Previously optional parameter is now mandatory for clarity."
```

---

## Learning More

- **Detailed release process:** See [RELEASE_PROCESS.md](../../RELEASE_PROCESS.md) in the repo root
- **Contribution basics:** See [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **Code style:** See [CLAUDE.md](../../CLAUDE.md)
- **Ask questions:** [GitHub Discussions](https://github.com/Data-Simply/pyretailscience/discussions)

---

## Quick Command Reference

```bash
# Setup
git clone https://github.com/Data-Simply/pyretailscience.git
cd pyretailscience
uv sync

# Create feature branch
git checkout -b feature/my-cool-feature main

# Run tests locally
uv run pytest
uv run pytest --cov=pyretailscience

# Lint and format
uv run ruff check .
uv run ruff format .

# Commit (conventional commits!)
git add .
git commit -m "feat: add my cool feature"

# Push and create PR
git push -u origin feature/my-cool-feature
gh pr create --base main

# After merge, update your local main
git checkout main
git pull origin main
```

---

**Welcome to the PyRetailScience community!** 🎉

Your contributions make this project better for everyone. If you have questions about the release process or anything else, don't hesitate to ask in GitHub Discussions or in your PR.

Happy coding! 🚀
