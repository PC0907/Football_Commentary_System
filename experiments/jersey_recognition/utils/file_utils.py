# utils/file_utils.py
import os
from pathlib import Path

def create_directory(path: str) -> Path:
    """Create directory if it doesn't exist"""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

def clean_directory(path: str) -> None:
    """Remove all files in a directory"""
    for file in Path(path).glob("*"):
        if file.is_file():
            file.unlink()
