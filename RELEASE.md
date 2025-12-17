# Release process

This repo uses tags for releases.

## Prereqs

- CI is green on `main`.
- `CHANGELOG.md` updated.
- Version bumped in `pyproject.toml` and `mq/__init__.py` (if applicable).

## Steps

1. Update `CHANGELOG.md` with the release version and date.
2. Bump version in `pyproject.toml`.
3. Run tests: `python -m unittest discover -s tests -p 'test*.py'`
4. Commit the release changes.
5. Tag and push:

   - `git tag -a vX.Y.Z -m "vX.Y.Z"`
   - `git push origin main --tags`
