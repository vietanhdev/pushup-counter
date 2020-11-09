import json
import cv2
import os
import numpy as np

INPUT_LABEL_FILE = "data/labels.json"
OUTPUT_LABEL_FILE = "data/labels-processed.json"
VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
HIT_PADDING = 10

with open(INPUT_LABEL_FILE, 'r') as infile:
    raw_data = json.load(infile)
    data = dict()

    labels = raw_data["result"]["labels"]
    labels.sort(key=lambda x:x["_id"])
    videos = {}
    for label in labels:
        content = json.loads(label["content"])
        if content["label"] is not None:
            videos[label["video_id"]] = {}
            video_path = os.path.join(VIDEO_FOLDER, "{}.mp4".format(label["video_id"]))

            if not os.path.isfile(video_path):
                print("Not found:{}".format(video_path))

            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            videos[label["video_id"]]["n_frames"] = frame_count
            hit_frames = [False] * (frame_count + 1)
            for l in content["label"]:
                for i in range(max(0, l-HIT_PADDING), min(frame_count-1, l+HIT_PADDING)):
                    hit_frames[i] = True

            final_label = []
            for i in range(len(hit_frames)):
                if hit_frames[i]:
                    final_label.append(i)
            videos[label["video_id"]]["label"] = final_label

    data["labels"] = videos
    with open(OUTPUT_LABEL_FILE, 'w') as outfile:
        json.dump(data, outfile)