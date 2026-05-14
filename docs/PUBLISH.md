# Publish runbook — PyPI + npm

Step-by-step for shipping the Python and JavaScript SDKs to package registries. Every command verified with the v0.7.3 build.

---

## PyPI (`pip install scm-memory`)

### Prerequisites (one-time)

1. PyPI account: https://pypi.org/account/register/
2. Generate an API token: https://pypi.org/manage/account/token/
   - Scope: "Entire account" for first publish, then narrow to `scm` after
3. Save the token in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIc...     # paste the full token starting with pypi-
```

(or use `keyring` / 1Password CLI for safer storage.)

### First-time publish

```bash
cd /Users/saish/Downloads/SleepAI
source venv/bin/activate

# Verify version is what you want
grep '^version' pyproject.toml

# Build (wheel + sdist)
rm -rf dist build *.egg-info
python -m build --wheel --sdist

# Validate
twine check dist/*

# (Optional) upload to TestPyPI first to verify
twine upload --repository testpypi dist/*
# Test install from TestPyPI:
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ scm

# Real publish
twine upload dist/*
```

After 30-60 seconds: `pip install scm-memory` works from anywhere in the world.

### Subsequent publishes

1. Bump version in `pyproject.toml`
2. Add a CHANGELOG entry
3. Same build + check + upload sequence as above

PyPI does NOT allow republishing the same version. Always bump.

### Verification

```bash
# In a fresh venv
python3.14 -m venv /tmp/scm-test && source /tmp/scm-test/bin/activate
pip install scm-memory
scm version
scm --help
```

---

## npm (`npm install scm-memory`)

### Prerequisites (one-time)

1. npm account: https://www.npmjs.com/signup
2. `npm login` once on your machine (persists in `~/.npmrc`)

### First-time publish

```bash
cd /Users/saish/Downloads/SleepAI/sdk/js

# Verify package.json metadata
cat package.json | head -30

# Dry run — see exactly what will be uploaded
npm publish --dry-run

# Real publish
npm publish --access=public
```

After ~1 minute: `npm install scm-memory` works.

### Subsequent publishes

```bash
cd /Users/saish/Downloads/SleepAI/sdk/js

# Bump version (patch / minor / major)
npm version patch  # 0.1.0 → 0.1.1

npm publish
```

### Verification

```bash
mkdir /tmp/npm-test && cd /tmp/npm-test
npm init -y
npm install scm-memory
node -e "const { SCM } = require('scm-memory'); console.log(new SCM())"
```

---

## Versioning policy

Both packages follow semver:

- **Patch** (0.7.3 → 0.7.4): bug fixes, internal refactors, no API change
- **Minor** (0.7.x → 0.8.0): new features; existing API stable
- **Major** (0.x → 1.0.0): API stability commitment

Keep `pyproject.toml` and `sdk/js/package.json` versions roughly in sync (independent bumps OK; don't let them drift more than one minor apart).

---

## Pre-publish checklist

Before pushing v0.7.x or higher to PyPI / npm:

- [ ] All 322 regression tests passing
- [ ] CHANGELOG.md updated with the new version
- [ ] README.md not embarrassingly out of date
- [ ] `LICENSE` file exists
- [ ] `pyproject.toml` version matches CHANGELOG
- [ ] `sdk/js/package.json` version matches CHANGELOG
- [ ] `python -m build && twine check dist/*` passes
- [ ] **Fresh-venv smoke test passes** (see below — `twine check` does NOT catch missing dependencies)
- [ ] `npm publish --dry-run` (in `sdk/js/`) shows the expected files
- [ ] Git tag created (e.g., `git tag v0.7.4 && git push --tags`) — optional but good practice

### Fresh-venv smoke test (MANDATORY — added after v0.7.3 hotfix)

`twine check` only validates packaging metadata. It does NOT verify that declared dependencies are sufficient. v0.7.3 shipped with three missing required deps (`sqlalchemy`, `rich`, `ollama`) — install succeeded but the very first import crashed. Always run:

```bash
# Build the wheel
python -m build --wheel

# Install into a CLEAN venv from the wheel only
rm -rf /tmp/scm-smoke && python3 -m venv /tmp/scm-smoke
/tmp/scm-smoke/bin/pip install dist/scm_memory-*.whl

# Smoke-test the public surface — run from /tmp so local src/ doesn't shadow
cd /tmp && /tmp/scm-smoke/bin/python -c "
from scm import SCMEngine, list_profiles
from src.integrations.langchain_adapter import SCMClient, SCMMemory
print('imports OK')
"

# Verify CLI entry point
/tmp/scm-smoke/bin/scm version
/tmp/scm-smoke/bin/scm --help | head -5
```

If any of those fail, **do not upload**. Add the missing dep to `pyproject.toml` `dependencies = [...]` and rebuild.

---

## Pre-launch sequence (when product-ready checklist completes)

The publish is ONE step in the launch sequence. The rest:

1. Make GitHub repo public
2. Push v0.7.x tag
3. **`twine upload dist/*`** (PyPI)
4. **`npm publish --access=public`** (npm)
5. arXiv paper push (from `research/arxiv_submission/scm-arxiv-bundle.tar.gz`)
6. Post the launch tweet / HN Show HN
7. Announce in relevant Discord/Slack channels

Steps 3-4 are what this runbook documents. Step 5 is gated separately (see `docs/ROADMAP.md`).

---

## Rollback

PyPI doesn't allow deletion of published versions (only "yanking" which keeps the version visible but flags it as bad). If you publish a broken version:

```bash
# Yank the broken version (still visible, but pip won't auto-install it)
twine upload --repository pypi --skip-existing dist/scm-0.7.3-py3-none-any.whl
# Then bump and publish a fix
```

npm allows `npm unpublish` within 72 hours of publish; after that, only deprecate:

```bash
npm unpublish scm-memory@0.1.0   # within 72h only
npm deprecate scm-memory@0.1.0 "Use 0.1.1 instead"
```

---

## Cost

Both PyPI and npm are free for public packages.
