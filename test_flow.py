import json
import os
from tokenize import single_quoted
import cv2
import numpy as np
from numpy import save
import matplotlib.pyplot as plt
import os
import io
import scipy
from scipy.signal import savgol_filter
import numpy
from tensorflow.python.keras.backend import sign

VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
OUTPUT_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/images"

FLOW_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/flow-112x112/"

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

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if window_len<3:
        return x


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

videos = os.listdir(VIDEO_FOLDER)
videos = [v for v in videos if v.endswith("mp4")]
skip = 1
flow_labels = {}
for i, video in enumerate(videos):
    print(i)
    cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret = True

    ret, img = cap.read()
    img = cv2.resize(img, (112, 112))

    print(frameCount)

    max_signal_len = 200
    signal = np.array([0] * max_signal_len)
    x = np.array(list(range(max_signal_len)), dtype=int)

    frame_id = 0

    signal = np.zeros((frameCount,), dtype=float)

    mask = None

    # prev_gray = None
    while (ret):
        img = None
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224))
            frame_id += 1

            tmp_mask = cv2.imread(os.path.join(
                FLOW_FOLDER, "{}_{}.png".format(video.split(".")[0], frame_id)))

            if tmp_mask is None:
                continue
            else:
                mask = tmp_mask

            # print(frame_id, "/", frameCount)

            angle = np.mean(mask[..., 0])
            signal[frame_id] = angle
            # if len(signal) > max_signal_len:
            #     signal = signal[1:]

            # signal = savgol_filter(signal, 5, 2)
            # signal = smooth(signal, window_len=5)
            # signal = signal[4:]

            # plot sin wave
            # fig = plt.figure()
            # ax = fig.add_subplot(111)
            # ax.plot(x, signal, "-r")
            # plot_img_np = get_img_from_fig(fig)
            

            # cv2.imshow("dense optical flow", plot_img_np)
            # cv2.imshow("dense", img)
            # cv2.waitKey(1)

    cap.release()
    signal = smooth(signal, window_len=10)
    signal = signal[9:]
    signal = smooth(signal, window_len=10)
    signal = signal[9:]
    signal = smooth(signal, window_len=10)
    signal = signal[9:]
    signal = signal[1:] - signal[:-1]
    signal = smooth(signal, window_len=20)
    signal = signal[19:]
    signal = np.append(signal, [0])
    # print(len(signal))
    bin_label = [0 if signal[i] > 0 else 1 for i in range(signal.shape[0])]
    flow_labels[video[:-4]] = bin_label
    # print(bin_label)
    
    # exit(1)

    # signal = (0.5 * signal[1:] + 0.5 * signal[:-1])

    # frame_id = 0
    # cap = cv2.VideoCapture(os.path.join(VIDEO_FOLDER, video))
    # ret, img = cap.read()
    # while (ret):
    #     img = None
    #     ret, img = cap.read()
    #     if ret:
    #         img = cv2.resize(img, (224, 224))
    #         frame_id += 1

    #         if frame_id > 200:

    #             if signal[frame_id] > 0:
    #                 img = cv2.rectangle(img, (10, 10), (50, 50), (0, 0, 255), -1) 

    #             fig = plt.figure()
    #             ax = fig.add_subplot(111)
    #             ax.plot(x, signal[frame_id-200:frame_id], "-r")
    #             plot_img_np = get_img_from_fig(fig)

    #             cv2.imshow("dense optical flow", plot_img_np)
    #             cv2.imshow("dense", img)
    #             cv2.waitKey(10)

with open("data/flow_labels.json", "w") as outfile:
    json.dump(flow_labels, outfile)

