"""Example game integration. Run from repository root."""
from kinectly import MotionEngine
from kinectly.events import EventBus
from kinectly.tracker import KinectlyTracker

engine, bus = MotionEngine(), EventBus()
bus.on('*', lambda event: print(f'{event.timestamp:6.2f}s {event.name} ({event.confidence:.2f})'))
with KinectlyTracker() as tracker:
    print('Tracking. Press Ctrl+C to exit.')
    try:
        while True:
            _, _, pose = tracker.read()
            for event in engine.update(pose):
                bus.publish(event)
    except KeyboardInterrupt:
        pass
