# Kinectly v0.2 — 3D Viewer

Experimental webcam-based body tracking with a live 2D overlay and interactive estimated 3D skeleton.

## Setup (Windows)

1. Copy `pose_landmarker_full.task` into the same folder as `main.py` (obtain from the official MediaPipe model distribution).
2. Install dependencies: `py -3.13 -m pip install -r requirements.txt`
3. Run: `py -3.13 main.py`

The webcam is opened with DirectShow at 640x480 to support the Logitech C920.

## Controls

- Left drag over 3D panel: rotate
- Mouse wheel over 3D panel: zoom
- R: reset camera view
- Q or Escape: quit

## Notes

The 3D points are *estimated* by MediaPipe from a single RGB camera. This is not a real depth sensor. This prototype is a viewer, not a trained action recognizer.

The webcam image is mirrored for display. The pose landmarks are inferred from the mirrored frame as well.
