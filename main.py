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
        self.ear_counter = []
        self.EAR_threshold = 0.2
        self.blink_count = 0
        self.img_canvas = tkinter.Canvas(
            self.window, width=self.vid_proc.width, height=self.vid_proc.height)
        self.img_canvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        # Gender Options
        self.g_opts = ["Male", "Female"]
        g_label = tkinter.Label(text="Select Gender")
        g_label.pack()
        self.gender_opt = tkinter.StringVar(self.window)
        self.gender_opt.set(self.g_opts[0])
        self.drop_gender = tkinter.OptionMenu(
            self.window, self.gender_opt, *self.g_opts)
        self.drop_gender.pack()
        # Age Options
        self.a_opts = ["1-20", "20-40", "40-60", "60-80", "80-100"]
        a_label = tkinter.Label(text="Select age group")
        a_label.pack()
        self.age_opt = tkinter.StringVar(self.window)
        self.age_opt.set(self.a_opts[0])
        self.age_drop = tkinter.OptionMenu(
            self.window, self.age_opt, *self.a_opts)
        self.age_drop.pack()
        # Time of day
        self.t_opts = ["morning", "afternoon", "evening", "night", "unknown"]
        t_label = tkinter.Label(text="Select Time of day")
        t_label.pack()
        self.tod_opt = tkinter.StringVar(self.window)
        self.tod_opt.set(self.t_opts[0])
        self.tod_drop = tkinter.OptionMenu(
            self.window, self.tod_opt, *self.t_opts)
        self.tod_drop.pack()
        # Plot
        self.fig, self.ax = plt.subplots()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(
            side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

        self.text_box = tkinter.Text(self.window, width=20)
        self.text_box.pack(side=tkinter.RIGHT)

        self.submit_btn = tkinter.Button(
            self.window, text="Submit", command=self.submit_opt)
        self.submit_btn.pack()
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

    def get_gender_opt(self, g_txt: str) -> int:
        return 0 if g_txt == "Male" else 1

    def get_age_opt(self, a_txt: str) -> int:
        if a_txt == "1-20":
            return 0
        elif a_txt == "20-40":
            return 1
        elif a_txt == "40-60":
            return 2
        elif a_txt == "60-80":
            return 3
        else:
            return 0

    def get_tod_opt(self, t_txt: str) -> int:
        if t_txt == "morning":
            return 0
        elif t_txt == "afternoon":
            return 1
        elif t_txt == "evening":
            return 2
        elif t_txt == "night":
            return 3
        else:
            return 4

    def submit_opt(self):
        gender = self.get_gender_opt(self.gender_opt.get())
        age = self.get_age_opt(self.age_opt.get())
        tod = self.get_tod_opt(self.tod_opt.get())
        print(f"Gender : {gender}\nAge : {age}\nTod : {tod}")


def main():
    root = tkinter.Tk()
    DfApp(root, "DeepFake Detection")


if __name__ == "__main__":
    main()
