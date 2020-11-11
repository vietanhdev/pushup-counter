import time
import json
import cv2
import numpy as np
from losses import focal_loss

video_path = "/mnt/DATA/PUSHUP_PROJECT/processed/395.mp4"
video_id = "395"

with open("data/labels-processed.json", "r") as infile:
    labels = json.load(infile)["labels"]
label = labels[video_id]["label"]


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
    if frame_id in label:
        img = cv2.rectangle(img, (10, 10), (50, 50), (0, 0, 255), -1) 
    img = cv2.resize(img, (500, 500))
    cv2.imshow("Result", img)
    cv2.waitKey(50)



