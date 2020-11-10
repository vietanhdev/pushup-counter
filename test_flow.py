import os
import cv2
import numpy as np
from numpy import save
import matplotlib.pyplot as plt
import os
import io
import scipy
from scipy.signal import savgol_filter

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


videos = os.listdir(VIDEO_FOLDER)
skip = 1
for i, video in enumerate(videos):

    print(i)
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret = True

    ret, img = cap.read()
    img = cv2.resize(img, (224, 224))
    mask = np.zeros_like(img)
    prev_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    max_signal_len = 200
    signal = np.array([0] * max_signal_len)
    x = np.array(list(range(max_signal_len)), dtype=int)

    # prev_gray = None
    while (ret):
        img = None
        ret, img = cap.read()
        ret, img = cap.read()
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Calculates dense optical flow by Farneback method
            # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowfarneback
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 50, 3, 5, 1.1, 0)

            # Computes the magnitude and angle of the 2D vectors
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # Sets image hue according to the optical flow direction
            mask[..., 0] = angle * 180 / np.pi / 2

            # Sets image value according to the optical flow magnitude (normalized)
            mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

            angle = np.mean(mask[..., 0])
            
            signal = np.append(signal, [angle])
            if len(signal) > max_signal_len:
                signal = signal[1:]

            # signal = savgol_filter(signal, 5, 2)


            # plot sin wave
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x, signal, "-r")
            plot_img_np = get_img_from_fig(fig)


            # Converts HSV to RGB (BGR) color representation
            rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
            
            # Updates previous frame
            prev_gray = gray

            cv2.imshow("dense optical flow", plot_img_np)
            cv2.imshow("gray", gray)
            cv2.waitKey(50)
