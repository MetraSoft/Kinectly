"""Kinectly v0.2 — webcam + interactive estimated-3D skeleton viewer.

Requires mediapipe Tasks, opencv-python, numpy and pose_landmarker_full.task.
"""
from __future__ import annotations
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

WIDTH, HEIGHT = 640, 480
MODEL = Path(__file__).resolve().parent / 'pose_landmarker_full.task'
CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
LEFT_COLOR = (90, 255, 130)
RIGHT_COLOR = (255, 200, 80)

class Viewer:
    def __init__(self):
        self.yaw = -0.25
        self.pitch = 0.10
        self.zoom = 165.0
        self.dragging = False
        self.last_xy = (0, 0)

    def mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and x >= WIDTH:
            self.dragging = True
            self.last_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx, dy = x - self.last_xy[0], y - self.last_xy[1]
            self.yaw += dx * 0.008
            self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.35, 1.35))
            self.last_xy = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL and x >= WIDTH:
            self.zoom = float(np.clip(self.zoom * (1.1 if flags > 0 else 1 / 1.1), 70, 360))

    def project(self, pts):
        # MediaPipe world coordinates: x right, y down, z depth; units ~metres.
        # Center on the hips so the body stays visible while moving.
        xyz = np.array([[p.x, -p.y, -p.z] for p in pts], dtype=np.float32)
        origin = (xyz[23] + xyz[24]) / 2
        xyz -= origin
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        rotated = xyz @ yaw.T @ pitch.T
        # Orthographic projection: stable size, easy to inspect joints.
        coords = np.column_stack((WIDTH // 2 + rotated[:, 0] * self.zoom,
                                  HEIGHT * 0.59 - rotated[:, 1] * self.zoom))
        return coords.astype(np.int32), rotated

    def render(self, pts):
        panel = np.full((HEIGHT, WIDTH, 3), (24, 24, 32), dtype=np.uint8)
        cv2.putText(panel, '3D SKELETON | ESTIMATED', (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 210, 115), 2, cv2.LINE_AA)
        cv2.putText(panel, 'Drag to rotate | Wheel to zoom | R reset', (18, HEIGHT - 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 190), 1, cv2.LINE_AA)
        # Subtle reference grid at the bottom of the panel.
        for i in range(8):
            cv2.line(panel, (0, 380 + i * 12), (WIDTH - 1, 380 + i * 12), (38, 38, 47), 1)
        if not pts:
            cv2.putText(panel, 'NO BODY DETECTED', (145, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 120, 255), 2)
            return panel
        xy, rotated = self.project(pts)
        for a, b in CONNECTIONS:
            if 0 <= a < len(xy) and 0 <= b < len(xy):
                color = LEFT_COLOR if a % 2 else RIGHT_COLOR
                cv2.line(panel, tuple(xy[a]), tuple(xy[b]), color, 2, cv2.LINE_AA)
        for i, (x, y) in enumerate(xy):
            if i in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                cv2.circle(panel, (int(x), int(y)), 4, (245, 245, 245), -1, cv2.LINE_AA)
        return panel


def draw_camera(frame, landmarks):
    if not landmarks:
        cv2.putText(frame, 'SEARCHING FOR BODY...', (18, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 170, 255), 2, cv2.LINE_AA)
        return
    xy = [(int(p.x * WIDTH), int(p.y * HEIGHT)) for p in landmarks]
    for a, b in CONNECTIONS:
        if a < len(xy) and b < len(xy):
            cv2.line(frame, xy[a], xy[b], (80, 245, 125), 2, cv2.LINE_AA)
    for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
        cv2.circle(frame, xy[i], 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, 'BODY TRACKED', (18, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 245, 125), 2, cv2.LINE_AA)


def main():
    if not MODEL.is_file():
        raise SystemExit(f'Missing model: {MODEL}\nCopy pose_landmarker_full.task into this folder.')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    if not cam.isOpened():
        raise SystemExit('Cannot open webcam index 0 with DirectShow.')
    viewer = Viewer()
    window = 'Kinectly v0.2 | 2D + 3D'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, WIDTH * 2, HEIGHT)
    cv2.setMouseCallback(window, viewer.mouse)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    start = time.monotonic()
    last_ts = -1
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cam.read()
                if not ok:
                    print('Camera read failed.')
                    break
                frame = cv2.resize(frame, (WIDTH, HEIGHT))
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
                timestamp = max(int((time.monotonic() - start) * 1000), last_ts + 1)
                last_ts = timestamp
                result = landmarker.detect_for_video(image, timestamp)
                pose = result.pose_landmarks[0] if result.pose_landmarks else None
                world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
                draw_camera(frame, pose)
                cv2.putText(frame, 'KINECTLY v0.2', (18, HEIGHT - 17),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                side = viewer.render(world)
                cv2.imshow(window, np.hstack((frame, side)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                if key == ord('r'):
                    viewer.yaw, viewer.pitch, viewer.zoom = -0.25, 0.10, 165.0
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
