import cv2
import tkinter
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from PIL import Image, ImageTk
from ear_utils import get_frame_EAR, VidProcess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DfApp:
    def __init__(self, window, title):
        self.window = window
        self.vid_proc = VidProcess("./vid_data/fake/0.mp4")
        self.window.title(title)
        self.image = ImageTk.PhotoImage(file="dfdetect.jpg")
        self.ear_counter: List[float] = []
        self.EAR_threshold = 0.2
        self.blink_count = 0
        self.img_canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.img_canvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

        self.fig, self.ax = plt.subplots()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(
            side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

        self.text_box = tkinter.Text(self.window, width=20)
        self.text_box.pack(side=tkinter.RIGHT)

        self.update()
        self.window.mainloop()

    def update(self):
        success, frame = self.vid_proc.get_frame()
        if not success:
            return
        frame, counter = get_frame_EAR(frame)
        if counter < self.EAR_threshold:
            self.blink_count += 1
        self.ear_counter.append(counter)

        self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.img_canvas.create_image(0, 0, image=self.image, anchor=tkinter.NW)

        self.ax.clear()
        self.ax.plot(self.ear_counter)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("EAR")
        self.plot_canvas.draw()

        self.text_box.delete("1.0", tkinter.END)
        self.text_box.insert(
            "1.0", chars=f"EAR : {counter}\nCounter : {self.blink_count}/{len(self.ear_counter)}")

        self.window.after(1, self.update)


def main():
    root = tkinter.Tk()
    DfApp(root, "DeepFake Detection")


if __name__ == "__main__":
    main()
