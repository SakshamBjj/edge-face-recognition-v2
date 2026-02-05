"""
Configuration loader.

Loads configuration from:
1) Packaged default.yaml (installed with the library)
2) A user-provided path via --config

Also resolves platform-dependent paths (OpenCV cascade).
"""

from pathlib import Path
import yaml
import cv2

try:  # Python 3.9+
    from importlib.resources import files, as_file
except ImportError:  # Python 3.8 fallback
    from importlib_resources import files, as_file


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def load_config(path: str | Path | None = None) -> dict:
    """
    Load YAML configuration.

    Parameters
    ----------
    path : str | Path | None
        Optional custom config path.
        - None → use packaged default
        - existing file → load user config
        - "default.yaml" → load packaged default

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """

    # No path → packaged default
    if path is None:
        return _load_packaged_default()

    path = Path(path)

    # Explicit user file
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return _resolve_paths(cfg)

    # Allow `--config default.yaml`
    if path.name == "default.yaml":
        return _load_packaged_default()

    raise FileNotFoundError(
        f"Config not found at '{path}'. "
        "Provide a valid file path or omit --config to use default settings."
    )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _load_packaged_default() -> dict:
    """Load default.yaml bundled inside the installed package."""
    resource = files("edge_face").joinpath("default.yaml")

    # as_file ensures a real filesystem path even if package is zipped
    with as_file(resource) as real_path:
        with open(real_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    return _resolve_paths(cfg)


def _resolve_paths(cfg: dict) -> dict:
    """
    Resolve runtime-dependent paths.

    Converts cascade filename → full OpenCV path.
    """
    cfg["face"]["cascade"] = cv2.data.haarcascades + cfg["face"]["cascade"]
    return cfg
