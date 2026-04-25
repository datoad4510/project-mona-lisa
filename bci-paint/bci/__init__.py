# bci/__init__.py
from .headset          import Headset
from .ssvep            import SSVEPDetector
from .color_classifier import ColorClassifier

__all__ = ["Headset", "SSVEPDetector", "ColorClassifier"]
