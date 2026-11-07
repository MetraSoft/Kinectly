"""Kinectly: webcam-based estimated 3D body tracking and motion events."""
from .motion import MotionEngine, MotionEvent, MotionConfig, PoseFrame, Joint
__version__ = '0.3.0-alpha'
__all__ = ['MotionEngine', 'MotionEvent', 'MotionConfig', 'PoseFrame', 'Joint']
