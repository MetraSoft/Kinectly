import unittest
from kinectly.motion import MotionEngine, PoseFrame, Joint, angle, distance

class MotionTests(unittest.TestCase):
    def test_angle(self):
        self.assertAlmostEqual(angle(Joint(1,0,0),Joint(0,0,0),Joint(0,1,0)),90)
    def test_distance(self):
        self.assertAlmostEqual(distance(Joint(0,0,0),Joint(0,3,4)),5)
    def test_missing_pose(self):
        self.assertEqual(MotionEngine().update(None),[])
    def test_stationary_no_punch(self):
        joints=[Joint(0,0,0) for _ in range(33)]
        joints[11]=Joint(-0.2,0,0); joints[12]=Joint(0.2,0,0)
        joints[23]=Joint(-0.2,0.5,0); joints[24]=Joint(0.2,0.5,0)
        joints[25]=Joint(-0.2,1.0,0); joints[26]=Joint(0.2,1.0,0)
        joints[27]=Joint(-0.2,1.5,0); joints[28]=Joint(0.2,1.5,0)
        engine=MotionEngine()
        names=[]
        for i in range(40):
            names.extend(e.name for e in engine.update(PoseFrame(i/30,joints)))
        self.assertFalse(any('punch' in n for n in names))

if __name__ == '__main__': unittest.main()
