from ear_utils import get_frame_EAR, VidProcess
from typing import Tuple, List
import numpy as np
import cv2


class DataProc:
    def __init__(self, file: str):
        self.vid_file = file
        self.vid_proc = VidProcess(self.vid_file)
        self.EAR_threshold = 0.2
        self.ear_counter = 0
        self.blink_count = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[float, bool]:
        success, frame = self.vid_proc.get_frame()
        if not success:
            raise StopIteration
        _, ear_val = get_frame_EAR(np.asarray(frame))
        return ear_val, ear_val < self.EAR_threshold

    def get_EAR_List(self) -> List[float]:
        return [res[0] for res in self]

    def get_blink_count(self) -> int:
        return len(list(filter(lambda res: res[1], iter(self))))

    def frame_count(self) -> int:
        return len(self)

    def avg_bpf(self) -> float:
        # blink / frame
        return (self.get_blink_count()/len(self))

    def __len__(self) -> int:
        return len(self.vid_proc)
