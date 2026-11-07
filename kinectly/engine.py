from collections import deque
from dataclasses import dataclass
from math import hypot

@dataclass(frozen=True)
class MotionEvent:
    name: str
    confidence: float = 1.0

class MotionEngine:
    """Lightweight body-relative gesture detection; no training required."""
    def __init__(self):
        self.history = deque(maxlen=12)
        self.cooldowns = {}
        self.frame = 0
        self.crouched = False
        self.arm_ready = {'left': False, 'right': False}

    def update(self, landmarks):
        """Accept 33 MediaPipe normalized landmarks; return a list of MotionEvent."""
        self.frame += 1
        if not landmarks:
            self.history.clear()
            self.arm_ready = {'left': False, 'right': False}
            return []
        pts = [(p.x, p.y, p.z, getattr(p, 'visibility', 1.0)) for p in landmarks]
        self.history.append(pts)
        events = []
        def emit(name, gap=12):
            if self.frame >= self.cooldowns.get(name, 0):
                events.append(MotionEvent(name))
                self.cooldowns[name] = self.frame + gap
        def d(a, b):
            return hypot(pts[a][0]-pts[b][0], pts[a][1]-pts[b][1])
        if any(pts[i][3] < .45 for i in (11,12,13,14,15,16,23,24)):
            return []
        shoulder_width = max(d(11,12), .08)
        for side, shoulder, elbow, wrist in [('left',11,13,15),('right',12,14,16)]:
            bent = d(elbow,wrist) > 0 and d(shoulder,wrist) < shoulder_width * .95
            extended = d(shoulder,wrist) > shoulder_width * 1.45
            if bent:
                self.arm_ready[side] = True
            if extended and self.arm_ready[side] and len(self.history) >= 4:
                old = self.history[-4]
                change = hypot(pts[wrist][0]-old[wrist][0],pts[wrist][1]-old[wrist][1])
                if change > shoulder_width * .45:
                    emit(f'{side}_arm_extend', 15)
                    self.arm_ready[side] = False
        torso = (d(11,23)+d(12,24))/2
        hip_y = (pts[23][1]+pts[24][1])/2
        shoulder_y = (pts[11][1]+pts[12][1])/2
        ratio = (hip_y-shoulder_y)/max(shoulder_width,.08)
        # Relative torso compression: works best with a mostly frontal view.
        if ratio < 0.85 and not self.crouched:
            emit('crouch', 20)
            self.crouched = True
        elif ratio > 1.05:
            self.crouched = False
        return events
