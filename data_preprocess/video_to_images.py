import os
import cv2
import numpy as np
from numpy import save
import pathlib

VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
OUTPUT_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/images-112x112"

pathlib.Path(OUTPUT_FOLDER).mkdir(exist_ok=True, parents=True)

videos = os.listdir(VIDEO_FOLDER)
skip = 1
for i, video in enumerate(videos):

    if not video.endswith("mp4"):
        continue

    print(i)
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # buf = np.empty((int(frameCount / skip), 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    frame_id = 0
    while (fc < frameCount  and ret):
        img = None
        for _ in range(skip):
            ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (112, 112))
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, "{}_{}.png".format(video[:-4], frame_id)), img)
            frame_id += 1
            # buf[int(fc / skip)] = img
        fc += 1

    # save(os.path.join(OUTPUT_FOLDER, "{}.npy".format(video[:-4])), buf)
    