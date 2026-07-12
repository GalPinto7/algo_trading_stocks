"""
Small service functions used by the API routes.

Keeping this logic outside the route file makes the API easier to test and
keeps Flask-specific code separate from project-data logic.
"""

from __future__ import annotations

from pathlib import Path

from paths import DATA_DIR

# todo: we need to keep track of this, maybe auto this
# a dict of all the models and ids we use
MODEL_CATALOG = {
    0: "Dumb baseline",
    1: "XGBoost",
    2: "Previous day return baseline",
    3: "Rolling average baseline",
    4: "Linear regression",
    5: "Fully connected neural network",
}


def get_model_catalog() -> list[dict]:
    """
    Return the model options supported by the project.

    Each item includes the integer flag used in the evaluation code and a
    human-readable model name for API consumers.
    """
    return [{"flag": flag, "name": model_name} for flag, model_name in MODEL_CATALOG.items()]


def get_dataset_inventory(data_dir: Path = DATA_DIR) -> dict:
    """
    Return lightweight metadata about files in the project's data folder.

    The function only reads file names, extensions, and sizes. It does not load
    large pickle files into memory, so it is safe for a quick API response.
    """
    files = []

    if data_dir.exists():
        for path in sorted(data_dir.iterdir()):
            if path.is_file():
                files.append({
                    "name": path.name,
                    "type": path.suffix.lstrip(".") or "unknown",
                    "size_bytes": path.stat().st_size,
                })

    return {
        "data_dir": str(data_dir),
        "file_count": len(files),
        "files": files,
    }
