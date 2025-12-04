# Repository Guidelines

## Project Structure & Modules
- Core package lives in `felo_extinction/` with submodules for modeling (`models/extinction_models.py`), Milky Way corrections (`mw_extinction/mw_extinction.py`), and scaffolding for fitting/data handling.
- Packaging metadata is in `setup.cfg`/`setup.py`; build config in `pyproject.toml`.
- Add new analysis utilities inside the relevant submodule; keep any example notebooks or figures out of the package tree (use `build/` or a top-level `notebooks/` you create).

## Setup, Build, and Development Commands
- Install for development: `python -m pip install -e .` (run inside the repo).
- Build a distributable wheel/sdist: `python -m pip install build` once, then `python -m build`.
- Lint/format (pep8-style): `python -m pip install black ruff` then `black .` and `ruff check .` before sending changes.
- Quick sanity check: open a Python shell and run `import felo_extinction; felo_extinction.__version__` to ensure the editable install works.

## Coding Style & Naming Conventions
- Follow PEP8: 4-space indents, snake_case for functions/variables, PascalCase for classes, and descriptive module names.
- Type hints encouraged for new functions; prefer explicit units when dealing with `astropy` quantities.
- Keep docstrings concise and include parameter units where relevant; prefer numpy-style docstrings for public APIs.

## Testing Guidelines
- Use `pytest` under a `tests/` directory (not currently present); mirror module paths (e.g., tests for `mw_extinction.py` in `tests/mw_extinction/test_mw_extinction.py`).
- Favor small, deterministic unit tests; when external data/maps are needed, mock the IO boundaries.
- Aim for coverage on new code paths and any bug fixes; add regression tests before changing model behavior.

## Commit & Pull Request Guidelines
- Commit messages should be short, imperative summaries (e.g., `add mw extinction helper`, matching the existing history).
- For pull requests, include: purpose and scope, key commands run (`python -m build`, `pytest`), and any data/model assumptions. Link to issues if applicable; add screenshots only when visual outputs are changed.
- Keep changes focused; separate refactors from behavior changes when possible.

## Security & Configuration Notes
- Do not commit data products, credentials, or cache outputs; the package should stay source-only.
- External catalogs (e.g., `dustmaps`) may require downloads—document paths and versions in PRs when they affect reproducibility.
