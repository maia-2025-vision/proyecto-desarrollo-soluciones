import json
import os
import subprocess
from hashlib import md5, sha256


def get_git_revision_hash() -> str:
    """Checks for pending Git changes and returns the current revision hash.

    Raises:
        Exception: If there are uncommitted changes in the current working directory.

    Returns:
        str: The full 40-character SHA-1 git revision hash.
    """
    try:
        # Check for uncommitted changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        if result.stdout and not os.environ.get("ALLOW_UNCOMMITED_FILES"):
            raise Exception("There are uncommitted changes in the current working directory.")

        # If clean, get the current revision hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except FileNotFoundError as exc:
        raise RuntimeError("Maybe Git is not installed or not in the system's PATH?") from exc
    except subprocess.CalledProcessError as e:
        raise Exception(f"An error occurred while running a git command: {e}") from e


def get_cfg_hash(cfg: dict[str, object] | str, length: int = 12) -> str:
    """Produce a hexadecimal hash of length of a cfg object, or str.

    cfg must be json serializable.
    """
    if not isinstance(cfg, str):
        cfg_str = json.dumps(cfg)
    else:  # already a str
        cfg_str = cfg

    return sha256(cfg_str.encode("utf8")).hexdigest()[:length]
