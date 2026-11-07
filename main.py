import time
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from kinectly.engine import MotionEngine

MODEL = Path(__file__).with_name('pose_landmarker_full.task')
if not MODEL.exists():
    raise SystemExit('Missing pose_landmarker_full.task beside main.py. Copy your existing MediaPipe model here.')

connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL)),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=.6,
    min_pose_presence_confidence=.6,
    min_tracking_confidence=.6,
)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not camera.isOpened():
    raise SystemExit('Could not open webcam')

engine = MotionEngine()
start = time.monotonic()
last_event = 'none'
try:
    with vision.PoseLandmarker.create_from_options(options) as tracker:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic()-start)*1000)
            result = tracker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w = frame.shape[:2]
                for a, b in connections:
                    p, q = landmarks[a], landmarks[b]
                    cv2.line(frame, (int(p.x*w),int(p.y*h)),(int(q.x*w),int(q.y*h)),(40,230,80),2)
                for event in engine.update(landmarks):
                    last_event = event.name
                    print('EVENT:', event.name)
            else:
                engine.update(None)
            cv2.putText(frame, 'KINECTLY v0.1 | ESC to quit', (12,28), cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
            cv2.putText(frame, 'Event: '+last_event, (12,56), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,255),2)
            cv2.imshow('Kinectly',frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
finally:
    camera.release()
    cv2.destroyAllWindows()
