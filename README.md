# Kinectly

Experimental open-source webcam-based body tracking and motion event engine by MetraSoft.

## Current features
- MediaPipe Tasks body tracking from a webcam
- Live 2D skeleton overlay
- Experimental arm-extension and crouch events
- Reusable `MotionEngine` Python class

**Note:** The current gestures are experimental and may produce false positives. 3D visualization and robust pretrained action recognition are not yet implemented.

## Setup (Windows)
Install Python and dependencies:

```cmd
py -3.13 -m pip install -r requirements.txt
```

Place your compatible MediaPipe `pose_landmarker_full.task` model in the repository root, then run:

```cmd
py -3.13 main.py
```

Uses DirectShow camera 0 at 640x480. Press ESC to quit.

## API

```python
from kinectly import MotionEngine
engine = MotionEngine()
# events = engine.update(mediapipe_pose_landmarks)
```

## License
Project code can be licensed under Apache-2.0 by its copyright holder. Third-party model assets retain their own licenses.
