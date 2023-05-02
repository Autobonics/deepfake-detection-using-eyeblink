import cv2
import tkinter
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Tuple, Optional
from PIL import Image, ImageTk
from xgboost import XGBClassifier
from ear_utils import get_frame_EAR, VidProcess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DfApp:
    def __init__(self, window, title):
        self.window = window
        self.vid_proc = None
        self.vid_path = None
        self.vid_flag = True
        self.window.title(title)
        self.image = ImageTk.PhotoImage(file="dfdetect.jpg")
        self.ear_counter = []
        self.EAR_threshold = 0.2
        self.blink_count = 0
        self.model = None
        self.res_str = "press submit after data selection and vid process"
        try:
            self.model = XGBClassifier()
            self.model.load_model('dfake_model.xgb')
        except Exception as err:
            print("Error getting model :", err)

        self.file_btn = tkinter.Button(
            text="Select Video", command=self.select_vid)
        self.file_btn.pack()
        self.vid_state = tkinter.StringVar()
        self.vid_label = tkinter.Label(
            self.window, textvariable=self.vid_state)
        self.vid_label.pack()
        self.canvas_width = 640
        self.canvas_height = 480
        self.img_canvas = tkinter.Canvas(
            self.window, width=self.canvas_width, height=self.canvas_height)
        self.img_canvas.pack(
            side=tkinter.TOP, fill=tkinter.BOTH, expand=True)
        # Gender Options
        self.g_opts = ["Male", "Female"]
        g_label = tkinter.Label(text="Select Gender")
        g_label.pack()
        self.gender_opt = tkinter.StringVar(self.window)
        self.drop_gender = tkinter.OptionMenu(
            self.window, self.gender_opt, *self.g_opts)
        self.drop_gender.pack()
        # Age Options
        self.a_opts = ["1-20", "20-40", "40-60", "60-80", "80-100"]
        a_label = tkinter.Label(text="Select age group")
        a_label.pack()
        self.age_opt = tkinter.StringVar(self.window)
        self.age_drop = tkinter.OptionMenu(
            self.window, self.age_opt, *self.a_opts)
        self.age_drop.pack()
        # Time of day
        self.t_opts = ["morning", "afternoon", "evening", "night", "unknown"]
        t_label = tkinter.Label(text="Select Time of day")
        t_label.pack()
        self.tod_opt = tkinter.StringVar(self.window)
        self.tod_drop = tkinter.OptionMenu(
            self.window, self.tod_opt, *self.t_opts)
        self.tod_drop.pack()
        # Submit button
        self.submit_btn = tkinter.Button(
            self.window, text="Submit", command=self.submit_opt)
        self.submit_btn.pack()
        # Plot
        self.fig, self.ax = plt.subplots()
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(
            side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)

        self.text_box = tkinter.Text(self.window, width=20)
        self.text_box.pack(side=tkinter.RIGHT)

        self.res_box = tkinter.Text(self.window)
        self.res_box.pack(side=tkinter.RIGHT)

        self.set_default()
        self.update()
        self.window.mainloop()

    def set_default(self):
        self.gender_opt.set(self.g_opts[0])
        self.age_opt.set(self.a_opts[0])
        self.tod_opt.set(self.t_opts[0])
        if self.vid_proc:
            self.vid_state.set("Processing Video")
        else:
            self.vid_state.set("Select Video to process")
        self.res_str = "Press Submit after data Selection and vid process"
        self.res_box.delete("1.0", tkinter.END)
        self.res_box.insert("1.0", chars=self.res_str)
        self.vid_flag = True
        self.ear_counter = []
        self.blink_count = 0

    def update(self):
        if self.vid_proc:
            success, frame = self.vid_proc.get_frame()
            if not success:
                self.vid_flag = False
                self.vid_state.set("Video Processing Completed")
                return
            frame, counter = get_frame_EAR(frame)
            if counter < self.EAR_threshold:
                self.blink_count += 1
            self.ear_counter.append(counter)

            self.image = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.img_canvas.create_image(
                0, 0, image=self.image, anchor=tkinter.NW)

            self.ax.clear()
            self.ax.plot(self.ear_counter)
            self.ax.set_xlabel("Frame")
            self.ax.set_ylabel("EAR")
            self.plot_canvas.draw()

            self.text_box.delete("1.0", tkinter.END)
            self.text_box.insert(
                "1.0", chars=f"EAR : {counter}\nCounter : {self.blink_count}/{len(self.ear_counter)}")
        self.window.after(1, self.update)

    def select_vid(self):
        self.vid_path = tkinter.filedialog.askopenfilename(
            initialdir="./", title="Select a video", filetypes=(("Mp4", "*.mp4"), ("all Files", "*.*")))
        self.vid_proc = VidProcess(self.vid_path)
        self.set_default()
        self.update()

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
        res = self.get_model_res(gender, age, tod)
        if res == 0:
            self.res_str = "Video is original"
        elif res == 1:
            self.res_str = "Video if fake"
        self.res_box.delete("1.0", tkinter.END)
        self.res_box.insert("1.0", chars=self.res_str)

    def get_model_res(self, gender: int, age: int, tod: int) -> Optional[int]:
        if self.model:
            if self.vid_flag == True:
                self.res_str = "Vid Processing not over"
                self.res_box.delete("1.0", tkinter.END)
                self.res_box.insert("1.0", chars=self.res_str)
                return None
            try:
                bpf = self.blink_count/len(self.vid_proc)
            except:
                bpf = 0
            print(f"Gender : {gender}\nAge : {age}\nTod : {tod}\nBpf: {bpf}")
            df = pd.DataFrame(data=[[age, gender, tod, bpf]],
                              columns=["age_grp", "gender", "time_of_day", "bpf"])
            try:
                res = self.model.predict(df)[0]
                return res
            except Exception as err:
                print("Error in model prediction : ", err)
                return None
        else:
            print("model not initialized")
            return None


def main():
    root = tkinter.Tk()
    DfApp(root, "DeepFake Detection")


if __name__ == "__main__":
    main()
