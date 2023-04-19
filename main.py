import cv2
import tkinter
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from PIL import Image, ImageTk
from ear_utils import get_frame_EAR
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class VidProcess:
    def __init__(self, vid_path: str):
        self.vid_path = vid_path
        self.cap = cv2.VideoCapture(vid_path)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

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
        self.ear_counter: List[float] = []
        self.img_canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.img_canvas.pack()

        self.fig, self.ax = plt.subplots()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack()

        self.update()
        self.window.mainloop()

    def update(self):
        success, frame = self.vid_proc.get_frame()
        if not success:
            return
        frame, counter = get_frame_EAR(frame)
        self.ear_counter.append(counter)

        self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.img_canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)

        self.ax.clear()
        self.ax.plot(self.ear_counter)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("EAR")
        self.plot_canvas.draw()

        self.window.after(1, self.update)


def main():
    root = tkinter.Tk()
    DfApp(root, "DeepFake Detection")


if __name__ == "__main__":
    main()
