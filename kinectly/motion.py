"""Lightweight, deterministic gesture heuristics. No training or recording required.

Coordinates are MediaPipe world landmarks (x right, y DOWN, z depth), in
estimated metres. These are not calibrated depth measurements.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Sequence

LS, RS, LE, RE, LW, RW = 11, 12, 13, 14, 15, 16
LH, RH, LK, RK, LA, RA = 23, 24, 25, 26, 27, 28

@dataclass(frozen=True)
class Joint:
    x: float
    y: float
    z: float
    visibility: float = 1.0

@dataclass(frozen=True)
class PoseFrame:
    timestamp: float
    joints: Sequence[Joint]

@dataclass(frozen=True)
class MotionEvent:
    name: str
    timestamp: float
    confidence: float = 0.6

@dataclass
class MotionConfig:
    min_visibility: float = 0.45
    crouch_enter: float = 0.76
    crouch_exit: float = 0.87
    arm_extension_angle: float = 155.0
    arm_ready_angle: float = 115.0
    punch_speed: float = 0.75
    punch_cooldown: float = 0.65
    run_min_alternations: int = 4
    run_window_seconds: float = 2.2
    jump_min_rise: float = 0.12


def distance(a: Joint, b: Joint) -> float:
    return sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)


def angle(a: Joint, pivot: Joint, b: Joint) -> float:
    u = (a.x-pivot.x, a.y-pivot.y, a.z-pivot.z)
    v = (b.x-pivot.x, b.y-pivot.y, b.z-pivot.z)
    dot = sum(x*y for x,y in zip(u,v))
    denom = sqrt(sum(x*x for x in u)*sum(y*y for y in v))
    if denom < 1e-8:
        return 0.0
    return degrees(acos(max(-1.0, min(1.0, dot/denom))))


class MotionEngine:
    """Update once per video frame. Returns discrete events; state is in .active.

    Heuristics are intentionally conservative. Punches are recognized as a
    bent-elbow-to-extended-elbow transition with wrist motion away from the
    shoulder, not merely any arm movement. Real-world performance varies.
    """
    def __init__(self, config: MotionConfig | None = None):
        self.config = config or MotionConfig()
        self.active: set[str] = set()
        self._arm_ready = {'left': False, 'right': False}
        self._last_punch = {'left': -999., 'right': -999.}
        self._previous: PoseFrame | None = None
        self._hip_baseline: float | None = None
        self._last_leg = 0
        self._steps = deque()
        self._last_run = -999.
        self._last_jump = -999.
        self._crouching = False
        self._airborne = False

    def reset(self):
        self.__init__(self.config)

    def update(self, frame: PoseFrame | None) -> list[MotionEvent]:
        if frame is None or len(frame.joints) < 33:
            self._previous = None
            self._arm_ready = {'left': False, 'right': False}
            self._steps.clear()
            self.active.clear()
            return []
        j, t, c = frame.joints, frame.timestamp, self.config
        if min(j[i].visibility for i in (LS,RS,LH,RH,LK,RK,LA,RA)) < c.min_visibility:
            self._previous = None
            self.active.clear()
            return []
        events: list[MotionEvent] = []
        def emit(name: str, conf: float = 0.65):
            events.append(MotionEvent(name, t, conf))
        torso = max(0.05, (distance(j[LS],j[LH])+distance(j[RS],j[RH]))/2)
        hip_y = (j[LH].y+j[RH].y)/2
        shoulder_y = (j[LS].y+j[RS].y)/2
        ankle_y = (j[LA].y+j[RA].y)/2
        height_ratio = (ankle_y-shoulder_y)/max(0.05, torso)
        # Calibrate from standing samples only, slowly; do not chase crouches.
        if self._hip_baseline is None:
            self._hip_baseline = hip_y
        elif not self._crouching and abs(hip_y-self._hip_baseline) < 0.08:
            self._hip_baseline = 0.99*self._hip_baseline + 0.01*hip_y
        # Crouch uses knee flexion plus reduced shoulder-to-ankle distance.
        knee_left = angle(j[LH],j[LK],j[LA])
        knee_right = angle(j[RH],j[RK],j[RA])
        knee_mean = (knee_left+knee_right)/2
        # Ratio based on torso length and shin length to accommodate different sizes.
        leg = (distance(j[LH],j[LA])+distance(j[RH],j[RA]))/2
        folded = (ankle_y-hip_y)/max(0.05, leg)
        is_crouch = knee_mean < (132 if not self._crouching else 151) and folded < (c.crouch_enter if not self._crouching else c.crouch_exit)
        if is_crouch and not self._crouching:
            emit('crouch_start', 0.75)
        elif not is_crouch and self._crouching:
            emit('crouch_end', 0.75)
        self._crouching = is_crouch
        # Alternating lifted knees; do not classify one knee lift as running.
        # Compare knee heights normalized to torso size.
        delta = (j[RK].y-j[LK].y)/torso
        leg_side = 1 if delta > 0.23 else -1 if delta < -0.23 else 0
        if leg_side and leg_side != self._last_leg and not is_crouch:
            self._steps.append((t, leg_side))
            self._last_leg = leg_side
        while self._steps and t-self._steps[0][0] > c.run_window_seconds:
            self._steps.popleft()
        running = len(self._steps) >= c.run_min_alternations and (t-self._steps[-1][0] < 0.7)
        if running and t-self._last_run > 1.0:
            emit('run', 0.6)
            self._last_run = t
        # Jump: estimate upward hip motion relative to previous hip baseline.
        # Camera pose estimation is not true global height; false positives possible.
        rise = self._hip_baseline - hip_y
        if rise > c.jump_min_rise and not self._airborne and not is_crouch and t-self._last_jump > 1.0:
            emit('jump', 0.5)
            self._last_jump = t
            self._airborne = True
        elif rise < c.jump_min_rise*0.35:
            self._airborne = False
        # Punch detection: bent elbow -> fast extension, wrist away from shoulder.
        for side, shoulder, elbow, wrist in (('left',LS,LE,LW),('right',RS,RE,RW)):
            if min(j[i].visibility for i in (shoulder,elbow,wrist)) < c.min_visibility:
                self._arm_ready[side] = False
                continue
            elbow_angle = angle(j[shoulder],j[elbow],j[wrist])
            if elbow_angle < c.arm_ready_angle:
                self._arm_ready[side] = True
            prev = self._previous
            if not prev or t-prev.timestamp <= 0 or t-prev.timestamp > 0.25:
                continue
            dt = t-prev.timestamp
            prev_w, prev_s = prev.joints[wrist], prev.joints[shoulder]
            prev_distance = distance(prev_w, prev_s)
            now_distance = distance(j[wrist], j[shoulder])
            extension_speed = (now_distance-prev_distance)/dt
            if (self._arm_ready[side] and elbow_angle >= c.arm_extension_angle
                and extension_speed > c.punch_speed and not running and not is_crouch
                and t-self._last_punch[side] >= c.punch_cooldown):
                height = 'high' if j[wrist].y < j[shoulder].y + 0.2*torso else 'low'
                emit(f'{side}_punch_{height}', 0.55)
                self._last_punch[side] = t
                self._arm_ready[side] = False
        self.active = ({'crouch'} if is_crouch else set()) | ({'run'} if running else set())
        self._previous = frame
        return events
