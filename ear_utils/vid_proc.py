import cv2
import numpy as np
from typing import Union, Tuple


class VidProcess:
    def __init__(self, vid_path: str):
        self.vid_path = vid_path
        self.cap = cv2.VideoCapture(vid_path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def __iter__(self):
        return self

    def get_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        success, image = False, None
        if self.cap.isOpened():
            success, image = self.cap.read()
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return success, image

    def __next__(self) -> Union[np.ndarray, None]:
        success, frame = self.get_frame()
        if not success:
            raise StopIteration
        return frame

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def __len__(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    