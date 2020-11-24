import sys
import pathlib
import subprocess
import torch
import pytorch_lightning as pl
import asteroid


def print_versions():
    """CLI function to get info about the Asteroid and dependency versions."""
    for k, v in asteroid_versions().items():
        print(f"{k:20s}{v}")


def asteroid_versions():
    return {
        "Asteroid": asteroid_version(),
        "PyTorch": pytorch_version(),
        "PyTorch-Lightning": pytorch_lightning_version(),
    }


def pytorch_version():
    return torch.__version__


def pytorch_lightning_version():
    return pl.__version__


def asteroid_version():
    asteroid_root = pathlib.Path(__file__).parent.parent.parent
    if asteroid_root.joinpath(".git").exists():
        return f"{asteroid.__version__}, Git checkout {get_git_version(asteroid_root)}"
    else:
        return asteroid.__version__


def get_git_version(root):
    def _git(*cmd):
        return subprocess.check_output(["git", *cmd], cwd=root).strip().decode("ascii", "ignore")

    try:
        commit = _git("rev-parse", "HEAD")
        branch = _git("rev-parse", "--symbolic-full-name", "--abbrev-ref", "HEAD")
        dirty = _git("status", "--porcelain")
    except Exception as err:
        print(f"Failed to get Git checkout info: {err}", file=sys.stderr)
        return ""
    s = commit[:12]
    if branch:
        s += f" ({branch})"
    if dirty:
        s += f", dirty tree"
    return s
