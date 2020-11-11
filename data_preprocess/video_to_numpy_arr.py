import os
import cv2
import numpy as np
from numpy import save

VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
OUTPUT_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/images"

videos = os.listdir(VIDEO_FOLDER)
skip = 1
for i, video in enumerate(videos):

    print(i)
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((int(frameCount / skip), 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        img = None
        for _ in range(skip):
            ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            buf[int(fc / skip)] = img
        fc += 1

    save(os.path.join(OUTPUT_FOLDER, "{}.npy".format(video[:-4])), buf)
    