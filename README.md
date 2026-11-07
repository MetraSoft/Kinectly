# Kinectly v0.3 alpha — by MetraSoft

**Kinect-style body tracking with a regular webcam.** This is an independent open-source project, not affiliated with or endorsed by Microsoft.

## Features
- Live 2D body overlay + rotatable estimated 3D skeleton side-by-side.
- 33 MediaPipe body landmarks and 3D world-coordinate adapter.
- Custom Python motion engine: crouch start/end, running-in-place, experimental jump and left/right high/low punch events.
- Event callbacks for Python games (`kinectly.events.EventBus`).
- Optional local JSONL landmark recorder for debugging (never enabled by default).
- Configurable thresholds, resettable motion engine, unit tests and an example integration.
- Windows DirectShow camera capture at 640×480, retaining the proven Logitech C920 configuration.

**Important:** Gesture classification is experimental, not a pretrained action model. It will make mistakes, especially with one-camera depth estimation and partial occlusion. No calibration/training recordings are required. The 3D skeleton is *estimated*, not measured with a depth camera. No footage is uploaded anywhere.

## Install (Windows 10, Python 3.13)

Copy the model `pose_landmarker_full.task` from your working Kinectly v0.2 folder to this folder. The model is not included in this ZIP. Review the model's terms before redistributing it.

```cmd
cd /d C:\Apps\Kinectly
py -3.13 -m pip install -r requirements.txt
py -3.13 main.py
```

**Controls:** drag over right-hand panel to rotate; mouse wheel to zoom; `R` resets view; `C` resets motion state; `Q` or Esc quits.

## Game API

```python
from kinectly import MotionEngine
engine = MotionEngine()
# Each frame, call engine.update(PoseFrame(timestamp_seconds, joints))
# and handle returned MotionEvent(name, timestamp, confidence) objects.
```

Run `py -3.13 examples/game_api.py` from the repository root to print live events without a GUI. Run unit tests with `py -3.13 -m unittest discover -s tests -v`.

## Structure

- `main.py`: live 2D+3D viewer and event HUD.
- `kinectly/motion.py`: gesture recognition heuristics and dataclasses.
- `kinectly/tracker.py`: reusable camera/MediaPipe wrapper.
- `kinectly/events.py`: event callback bus.
- `kinectly/recording.py`: opt-in local landmark capture.
- `examples/game_api.py`: event integration demo.
- `tests/`: synthetic unit tests.

## Roadmap

Pretrained skeleton action models, automatic body calibration, stronger gesture state machines, multiple players, game controller mappings, latency benchmarks, and a game demo are **future work**, not current features.

## License

Project code is intended for Apache-2.0; preserve your existing repository's `LICENSE` file when copying these files. Third-party MediaPipe models and dependencies retain their own licenses.
