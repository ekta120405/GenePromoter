"""
Repo-root-relative paths, independent of whatever directory a script is
invoked from. train.py/evaluate.py/predict.py used to hardcode paths like
"data/train_1000bp.csv" assuming they'd always be run from the repo root --
which broke repeatedly when a notebook cell `%cd src` first (or when any
other cwd is in play, e.g. a Docker container for the CC handoff).
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    """Join parts onto the repo root. An absolute part short-circuits, same as os.path.join."""
    return os.path.join(REPO_ROOT, *parts)
