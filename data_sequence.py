import json
from tensorflow.keras.utils import Sequence
import math
import os
import time
import cv2
import numpy as np

class DataSequence(Sequence):

    def __init__(self, video_folder, label_file, batch_size, seq_len=10, y_steps=1):
        self.batch_size = batch_size
        self.video_folder = video_folder
        with open(label_file, "r") as infile:
            labels = json.load(infile)["labels"]

        self.time_steps = 2
        self.y_steps = y_steps

        # Only for 1 video
        # if "270" in labels:
        #     labels = {"270": labels["270"]}

        self.seq_len = seq_len
        self.batch_size_times_seq_len = self.batch_size * self.seq_len

        frames = []
        for video_id, label in labels.items():
            for i in range(0, (label["n_frames"] // self.batch_size_times_seq_len) * self.batch_size_times_seq_len, self.time_steps):
                frames.append({
                    "video_id": video_id,
                    "frame_id": i,
                    "label": 1 if i in label["label"] else 0
                })

        self.frames = frames
        self.n_samples = len(frames)

        # Show frames
        # for frame in frames:
        #     img_path =  os.path.join(
        #         self.video_folder, "{}_{}.png".format(frame["video_id"], frame["frame_id"]))
        #     img = cv2.imread(img_path)

        #     if frame["label"] == 1:
        #         img = cv2.rectangle(img, (10, 10), (50, 50), (0, 0, 255), -1) 

        #     cv2.imshow("Debug", img)
        #     cv2.waitKey(30)


    def __len__(self):
        return math.floor(self.n_samples / self.batch_size_times_seq_len)

    def __getitem__(self, idx):

        start_id = idx * self.batch_size_times_seq_len
        end_id = (idx + 1) * self.batch_size_times_seq_len
        frame_seq = self.frames[start_id:end_id] 
        first_frame = frame_seq[0]

        # Process the gap between 2 videos
        # => Ensure frames come from 1 video
        if any(f["video_id"] != first_frame["video_id"] for f in frame_seq):
            frame_seq = [first_frame] * self.batch_size_times_seq_len
        video_id = first_frame["video_id"]
    
        batch_x = []
        for f in frame_seq:
            img_path =  os.path.join(
                self.video_folder, "{}_{}.png".format(video_id, f["frame_id"]))
            img = cv2.imread(img_path)
            if img is None:
                print(img_path)
                exit(1)
            img = img - 127.5
            img /=  127.5
            batch_x.append(img)
        batch_x = np.array(batch_x)
        batch_y = np.array([frame_seq[i]["label"] for i in range(0, len(frame_seq), self.y_steps)])

        if self.seq_len != 1:
            batch_x = batch_x.reshape((self.batch_size, self.seq_len, 112, 112, 3))
            batch_y = batch_y.reshape((self.batch_size, self.seq_len // self.y_steps, 1))
        else:
            batch_x = batch_x.reshape((self.batch_size, 112, 112, 3))
            batch_y = batch_y.reshape((self.batch_size, 1))

        return batch_x, batch_y