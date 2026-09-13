"""Vision package for camera handling and CNN inference."""
from .camera import CameraManager
from .classifier import AnimalClassifier

__all__ = ["CameraManager", "AnimalClassifier"]
