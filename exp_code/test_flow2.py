import os
import cv2
import numpy as np
from numpy import save
import matplotlib.pyplot as plt
import os
import io
import cv2 as cv
import math

VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
OUTPUT_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/images"

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


videos = os.listdir(VIDEO_FOLDER)
skip = 1
for i, video in enumerate(videos):

    print(i)
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret = True

        # Create some random colors
    color = np.random.randint(0,255,(100,3))
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    
    max_signal_len = 200
    signal = [0] * max_signal_len
    x = np.array(list(range(max_signal_len)), dtype=int)

    # prev_gray = None
    while (ret):
        ret,frame = cap.read()
        ret,frame = cap.read()
        ret,frame = cap.read()
        ret,frame = cap.read()
        ret,frame = cap.read()
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        angles = []
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            angles.append(math.degrees(math.atan2(a-c, b-d)))

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        angle = np.mean(np.array(angles))
        
        signal.append(angle)
        if len(signal) > max_signal_len:
            signal.remove(signal[0])


        # plot sin wave
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x, signal, "-r")
        plot_img_np = get_img_from_fig(fig)


        cv2.imshow("dense optical flow", plot_img_np)
        cv2.imshow("gray", frame)
        cv2.waitKey(50)
