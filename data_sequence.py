import json
from tensorflow.keras.utils import Sequence
import math
import os
import time
import numpy as np

class DataSequence(Sequence):

    def __init__(self, video_folder, label_file, batch_size):
        self.batch_size = batch_size
        self.video_folder = video_folder
        labels = json.loads(label_file)["labels"]
        frames = []
        for video_id, label in enumerate(labels):
            for i in range(len(label["n_frames"])):
                frames.append({
                    "video_id": video_id,
                    "frame_id": i,
                    "label": 1 if i in label["label"] else 0
                })
        self.frames = frames
        self.n_samples = len(frames)
        self.videos = {}

    def __len__(self):
        return math.ceil(self.n_samples / self.batch_size)

    def __getitem__(self, idx):

        start_id = idx * self.batch_size
        end_id = (idx + 1) * self.batch_size
        frame_seq = self.frames[start_id:end_id] 
        first_frame = self.frame_seq[0]

        # Process the gap between 2 videos
        # => Ensure frames come from 1 video
        if any(f["video_id"] != first_frame["video_id"] for f in frame_seq):
            frame_seq = [first_frame] * self.batch_size

        video_id = first_frame["video_id"]
        if video_id not in self.videos:
            self.videos[video_id] = {
                "data": np.load(os.path.join(self.video_folder,
                "{}.npy".format(video_id))),
                "load_time": time.time()
            }

        # Delete video from cache
        if len(self.videos) > 15:
            for v in self.videos.keys():
                if time.time() - self.videos[v]["load_time"] > 5 * 60:
                    self.videos.pop(v, None)
    
        batch_x = np.array([f for f in ])
        batch_y = np.array([f["label"] for f in frame_seq])

        return np.array([
            resize(imread(file_name), (200, 200))
               for file_name in batch_x]), np.array(batch_y)