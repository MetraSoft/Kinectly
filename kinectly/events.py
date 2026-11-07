"""Minimal callback event bus for game integrations."""
from collections import defaultdict

class EventBus:
    def __init__(self): self._handlers = defaultdict(list)
    def on(self, event_name, callback):
        self._handlers[event_name].append(callback)
        return callback
    def publish(self, event):
        for callback in self._handlers[event.name] + self._handlers['*']:
            callback(event)
