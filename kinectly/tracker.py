"""MediaPipe Tasks wrapper; imports are deferred so unit tests need no camera."""
from __future__ import annotations
from pathlib import Path
import time
from .motion import Joint, PoseFrame

class KinectlyTracker:
    def __init__(self, model='pose_landmarker_full.task', camera_index=0, width=640, height=480):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python, vision
        self.cv2, self.mp = cv2, mp
        self.width, self.height = width, height
        path = Path(model)
        if not path.exists():
            raise FileNotFoundError(f'Missing MediaPipe model: {path}')
        self.cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cam.isOpened():
            raise RuntimeError('Could not open camera using DirectShow')
        opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6, min_tracking_confidence=0.6)
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)
        self.start = time.monotonic()
        self.last_ms = -1

    def read(self):
        import numpy as np
        ok, frame = self.cam.read()
        if not ok:
            return None, None, None
        cv2 = self.cv2
        frame = cv2.flip(cv2.resize(frame, (self.width, self.height)), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ms = max(int((time.monotonic()-self.start)*1000), self.last_ms+1)
        self.last_ms = ms
        result = self.landmarker.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ms)
        pose = result.pose_landmarks[0] if result.pose_landmarks else None
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
        pf = PoseFrame(ms/1000., [Joint(p.x,p.y,p.z,getattr(p,'visibility',1.)) for p in world]) if world else None
        return frame, pose, pf

    def close(self):
        self.landmarker.close()
        self.cam.release()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
