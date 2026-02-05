"""
Configuration loader.
Reads configs/default.yaml and resolves platform-dependent paths at runtime.
"""

from pathlib import Path
import yaml
import cv2


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    """Load YAML config. Raises FileNotFoundError with a clear message if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found at '{path}'. "
            "Make sure you're running from the repo root."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve cascade to full filesystem path at runtime.
    # cv2.data.haarcascades gives the platform-correct prefix
    # (e.g. /usr/local/lib/python3.x/site-packages/cv2/data/ on Linux,
    #  different path on Windows/Mac). The YAML only stores the filename.
    cfg["face"]["cascade"] = cv2.data.haarcascades + cfg["face"]["cascade"]

    return cfg
