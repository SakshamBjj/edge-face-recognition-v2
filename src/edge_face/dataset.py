"""
Dataset persistence layer.
All pickle I/O lives here — load, save, and append logic in one place.
Filenames match the existing add_faces.py / test.py convention
(faces_data.pkl, names.pkl) so data already on disk stays compatible.
"""

from pathlib import Path
import pickle
import numpy as np


class FaceDataset:
    def __init__(self, data_dir: str | Path):
        self.root = Path(data_dir)
        self.faces_path = self.root / "faces_data.pkl"
        self.names_path = self.root / "names.pkl"

    @property
    def exists(self) -> bool:
        return self.faces_path.exists() and self.names_path.exists()

    def load(self) -> tuple[np.ndarray, list[str]]:
        """Load and return (X, y). Raises FileNotFoundError if data is missing."""
        if not self.exists:
            raise FileNotFoundError(
                f"No training data in '{self.root}'.\n"
                "Run:  edge-face collect --name <YourName>\n"
                "to capture faces first."
            )
        with open(self.faces_path, "rb") as f:
            X = pickle.load(f)
        with open(self.names_path, "rb") as f:
            y = pickle.load(f)
        
        # Validation guards: fail fast on corrupt data
        if len(X) == 0:
            raise ValueError(
                f"Dataset is empty (0 samples). "
                f"'{self.faces_path}' may be corrupted."
            )
        if len(X) != len(y):
            raise ValueError(
                f"Data corruption: {len(X)} face samples but {len(y)} labels. "
                f"Check '{self.faces_path}' and '{self.names_path}'."
            )
        if X.ndim != 2:
            raise ValueError(
                f"Expected 2D face array (n_samples, features), got shape {X.shape}. "
                f"'{self.faces_path}' may be corrupted."
            )
        
        return X, y

    def append(self, new_faces: np.ndarray, name: str):
        """Append new face samples for one person. Creates files if first run."""
        self.root.mkdir(parents=True, exist_ok=True)

        new_labels = [name] * len(new_faces)

        if self.exists:
            X, y = self.load()
            X = np.vstack([X, new_faces])
            y = y + new_labels
            print(f"[INFO] Appended to existing data. Total samples: {len(X)}")
        else:
            X = new_faces
            y = new_labels
            print(f"[INFO] Created new dataset. Total samples: {len(X)}")

        with open(self.faces_path, "wb") as f:
            pickle.dump(X, f, protocol=4)
        with open(self.names_path, "wb") as f:
            pickle.dump(y, f, protocol=4)

        print(f"[SUCCESS] Saved to '{self.root}'")
