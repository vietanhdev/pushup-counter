import time
import json
import cv2
import numpy as np
from utils import download_file, read_label_file
import pathlib
import os


label_url = "https://vinbdi-label.herokuapp.com/labels?password=55251525"
video_url_pattern = "https://pushup.imfast.io/{}.mp4"
video_folder = "videos"
video_id = "257"

# Download files
print("Downloading labels")
download_file(label_url, "labels.json")
with open("labels.json", "r") as infile:
    labels = json.load(infile)
labels = read_label_file(label_file="labels.json")
print("Total video number:", len(labels))
trigger_frames = labels[video_id]["label"]

# Download video file
print("Downloading video")
pathlib.Path(video_folder).mkdir(exist_ok=True, parents=True)
video_path = os.path.join(video_folder, "{}.mp4".format(video_id))
video_url = video_url_pattern.format(video_id)
if not os.path.isfile(video_path):
    download_file(video_url, video_path)

cap = cv2.VideoCapture(video_path)
count = 0
current_value = 0
frame_id = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    frame_id += 1
    img = cv2.resize(img, (224, 224))
    padding = 10
    for f in trigger_frames:
        if frame_id > f - 5 and frame_id < f + 5:
            img = cv2.rectangle(img, (10, 10), (50, 50), (0, 0, 255), -1) 
    img = cv2.resize(img, (500, 500))
    cv2.imshow("Result", img)
    cv2.waitKey(30)



