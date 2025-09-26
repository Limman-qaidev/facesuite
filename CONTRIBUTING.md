# Contributing to FaceSuite


Thanks for considering a contribution!


## Setup
1. Create a virtualenv and install dependencies:
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```
2. (Optional) Install dev tools:
```bash
pip install black isort ruff pre-commit
pre-commit install
```


## Guidelines
- Keep commits small and meaningful.
- Follow PEP 8 and add clear docstrings.
- When changing behavior, update README and add tests (when available).


## Pull Requests
- Describe the motivation and changes.
- Include before/after or a short demo where relevant.