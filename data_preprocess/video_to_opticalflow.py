import os
import cv2
import numpy as np
from numpy import save
import pathlib
import cv2 as cv

VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
OUTPUT_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/flow-112x112"

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
    ret, img = cap.read()
    img = cv2.resize(img, (112, 112))
    

    # Creates an image filled with zero 
    # intensities with the same dimensions  
    # as the frame 
    mask = np.zeros_like(img) 
    
    # Sets image saturation to maximum 
    mask[..., 1] = 255

    prev_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    fc = 0
    ret = True

    frame_id = 0
    while (fc < frameCount  and ret):
        img = None
        for _ in range(skip):
            ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (112, 112))
            # Converts each frame to grayscale - we previously  
            # only converted the first frame to grayscale 
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            
            # Calculates dense optical flow by Farneback method 
            flow = cv.calcOpticalFlowFarneback(prev_gray, gray,  
                                            None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0) 
            
            # Computes the magnitude and angle of the 2D vectors 
            magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1]) 
            
            # Sets image hue according to the optical flow  
            # direction 
            mask[..., 0] = angle * 180 / np.pi / 2
            
            # Sets image value according to the optical flow 
            # magnitude (normalized) 
            mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX) 
            
            # Converts HSV to RGB (BGR) color representation 
            # rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR) 
            
            # Opens a new window and displays the output frame 
            # cv.imshow("dense optical flow", mask) 
            # cv.waitKey(10)
            
            # Updates previous frame 
            prev_gray = gray 
            # img = cv2.resize(img, (112, 112))
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, "{}_{}.png".format(video[:-4], frame_id)), mask)
            frame_id += 1
            # buf[int(fc / skip)] = img
        fc += 1

    # save(os.path.join(OUTPUT_FOLDER, "{}.npy".format(video[:-4])), buf)
    