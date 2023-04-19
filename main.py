import cv2
import tkinter
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from PIL import Image, ImageTk
from ear_utils import get_frame_EAR

Landmark = List[np.ndarray]


class VidProcess:
    def __init__(self, vid_path: str):
        self.vid_path = vid_path
        self.cap = cv2.VideoCapture(vid_path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.ear_counter: List[float] = []

    def get_frame(self) -> Tuple[bool, Union[np.ndarray, None]]:
        success, image = False, None
        if self.cap.isOpened():
            success, image = self.cap.read()

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return success, image

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class DfApp:
    def __init__(self, window, title):
        self.window = window
        self.vid_proc = VidProcess("./vid_data/fake/2.mp4")
        self.window.title(title)
        self.image = ImageTk.PhotoImage(file="dfdetect.jpg")
        self.canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.canvas.pack()
        self.text_label = tkinter.Label(self.window, text="Res text : ")
        self.text_label.pack()
        self.text_box = tkinter.Text(self.window, height=10)
        self.text_box.pack()
        self.update()
        self.window.mainloop()

    def update(self):
        success, frame = self.vid_proc.get_frame()
        if not success:
            return
        frame, counter = get_frame_EAR(frame)
        self.ear_counter.append(counter)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)
        self.window.after(1, self.update)


def main():
    root = tkinter.Tk()
    DfApp(root, "DeepFake Detection")


if __name__ == "__main__":
    main()
