# common

A Poetry metapackage that centralises Python dependencies shared by `backend/` and `desktop/`. It has no source code of its own — only a `pyproject.toml` pinning common tooling (pytest, setuptools, wheel) so that every downstream module resolves the same versions.

## Why it exists

Keeping baseline dependencies in one place avoids the "each subproject pins pytest differently" drift. Downstream modules consume it via a local path dependency in their own `pyproject.toml`, e.g.:

```toml
[tool.poetry.dependencies]
common-deps = { path = "../common", develop = true }
```

## Structure

```
common/
├── pyproject.toml    # name = "common-deps"; shared dependency pins
├── poetry.lock
└── src/              # placeholder package (empty)
    └── __init__.py
```

## Installing

The package is consumed transitively when you run `poetry install` in a downstream module (`backend/`, `desktop/`). You normally don't install it directly. To refresh the lockfile after editing `pyproject.toml`:

```bash
cd common
poetry lock --no-update
```

## Adding a shared dependency

1. Add the package under `[tool.poetry.dependencies]` in `common/pyproject.toml`.
2. Run `poetry lock` inside `common/`.
3. Run `poetry install` in each downstream module that needs the refresh.
