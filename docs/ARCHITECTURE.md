# Kinectly architecture

`Webcam (DirectShow) -> MediaPipe PoseLandmarker -> 33 2D/estimated 3D landmarks -> MotionEngine -> MotionEvent -> Game`.

Viewer and gesture recognition are separate. The motion engine only consumes timestamped joint arrays and does not import OpenCV or MediaPipe. That makes synthetic tests and future trackers possible. Gestures are conservative heuristics rather than trained classifiers; this distinction matters when comparing to a Kinect sensor or pretrained action-recognition model.

### Privacy

No cloud calls or uploads are performed by Kinectly. `LandmarkRecorder` writes only if a developer explicitly instantiates it. Treat recorded skeleton sequences as personal data; do not commit recordings.

### Current limitations

Single-person, single-camera; estimated depth may drift; arm occlusion, side views, camera motion, and running motions can trigger false classifications. The current punch classification is a game input gesture only, not an evaluation of real-world technique. Tune thresholds in `MotionConfig` after observing performance.
