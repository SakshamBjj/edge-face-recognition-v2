"""
Configuration loader.
Loads from bundled default.yaml (package resource) or user-specified path.
Resolves platform-dependent paths at runtime.
"""

from pathlib import Path
import yaml
import cv2

try:
    # Python 3.9+
    from importlib.resources import files
except ImportError:
    # Python 3.7-3.8 fallback
    from importlib_resources import files


def load_config(path: str | Path | None = None) -> dict:
    """Load YAML config from bundled resource or user path.
    
    Args:
        path: Custom config path. If None, loads bundled default.yaml
    
    Returns:
        dict: Parsed configuration
        
    Raises:
        FileNotFoundError: If custom path specified but doesn't exist
    """
    if path is not None:
        # User specified custom config
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Config not found at '{path}'. "
                "Check the path or omit --config to use default."
            )
        with open(path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        # Load bundled default config from package resources
        # This works both in development (editable install) and after pip install
        config_resource = files("edge_face").joinpath("default.yaml")
        with config_resource.open("r") as f:
            cfg = yaml.safe_load(f)

    # Resolve cascade to full filesystem path at runtime.
    # cv2.data.haarcascades gives the platform-correct prefix
    # (e.g. /usr/local/lib/python3.x/site-packages/cv2/data/ on Linux,
    #  different path on Windows/Mac). The YAML only stores the filename.
    cfg["face"]["cascade"] = cv2.data.haarcascades + cfg["face"]["cascade"]

    return cfg