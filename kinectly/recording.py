"""Opt-in local JSONL landmark recording; OFF unless explicitly enabled."""
import json
from dataclasses import asdict
from .motion import PoseFrame

class LandmarkRecorder:
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
    def write(self, frame: PoseFrame):
        self.file.write(json.dumps(asdict(frame), separators=(',',':'))+'\n')
    def close(self): self.file.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
