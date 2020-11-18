import json
from tensorflow.keras.utils import Sequence
import math
import os
import time
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

class DataSequence(Sequence):

    def __init__(self, images_folder, flow_folder, label_file, seq_len=10, batch_size=1):

        self.batch_size = batch_size
        self.images_folder = images_folder
        self.flow_folder = flow_folder
        self.seq_len = seq_len
        self.batch_size_times_seq_len = self.batch_size * self.seq_len
        with open(label_file, "r") as infile:
            data = json.load(infile)

        self.signals = []
        self.labels = []
        video_ids = []
        for video, video_data in data.items():
            # print(video)
            for i in range(0, len(video_data["signal"]), self.batch_size_times_seq_len):
                print(video_data["signal"][i], "-", video_data["label"][i])
                self.signals.append(video_data["signal"][i])
                self.labels.append(video_data["label"][i])
                video_ids.append(video)

        # self.image_names = []
        # self.labels = []
        # for l in self.labels:
        #     image_name = "{}_{}".format()
        #     flow_path =  os.path.join(self.flow_folder, image_name + ".png")
        #     img_path =  os.path.join(self.images_folder, image_name + ".png")
        #     if os.path.isfile(flow_path) and os.path.isfile(img_path):
        #         self.image_names.append(image_name)

        # self.image_names =  self.image_names[:20000]
        # self.labels =  self.labels[:20000]

        print(len(self.signals), self.batch_size_times_seq_len)
        self.n_samples = len(self.signals)
        self.n_batches = math.floor(self.n_samples / self.batch_size_times_seq_len)


    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):

        # print(self.n_samples)
        # exit(1)

        start_id = idx * self.batch_size_times_seq_len
        end_id = (idx + 1) * self.batch_size_times_seq_len
        # batch_image_names = self.image_names[start_id:end_id] 

        # batch_img = []
        # for image_name in batch_image_names:
        #     img_path =  os.path.join(self.images_folder, image_name + ".png")
        #     img = cv2.imread(img_path)
        #     if img is None:
        #         print(img_path)
        #         exit(1)
        #     batch_img.append(img)
        # batch_img = np.array(batch_img)


        # ia.seed(1)
        # seq = iaa.Sequential([
        #     # Small gaussian blur with random sigma between 0 and 0.5.
        #     # But we only blur about 50% of all images.
        #     iaa.Sometimes(
        #         0.5,
        #         iaa.GaussianBlur(sigma=(0, 0.5))
        #     ),
        #     # Strengthen or weaken the contrast in each image.
        #     iaa.LinearContrast((0.75, 1.5)),
        #     # Add gaussian noise.
        #     # For 50% of all images, we sample the noise once per pixel.
        #     # For the other 50% of all images, we sample the noise per pixel AND
        #     # channel. This can change the color (not only brightness) of the
        #     # pixels.
        #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        #     # Make some images brighter and some darker.
        #     # In 20% of all cases, we sample the multiplier once per channel,
        #     # which can end up changing the color of the images.
        #     iaa.Multiply((0.8, 1.2), per_channel=0.2)
        # ], random_order=True) # apply augmenters in random order
        # batch_img = seq(images=batch_img)

        # batch_img = batch_img - 127.5
        # batch_img /=  127.5

        # # for img in batch_img:
        # #     cv2.imshow("img", img)
        # #     cv2.waitKey(0)

        # batch_flow = []
        # for image_name in batch_image_names:
        #     flow_path =  os.path.join(self.flow_folder, image_name + ".png")
        #     flow_img = cv2.imread(flow_path)
        #     if flow_img is None:
        #         print(flow_path)
        #         exit(1)
        #     batch_flow.append(flow_img)
        # batch_flow = np.array(batch_flow)

        # batch_y = np.array(self.labels[start_id:end_id])
        # if self.seq_len != 1:
        #     batch_img = batch_img.reshape((self.batch_size, self.seq_len, 112, 112, 3))
        #     batch_flow = batch_flow.reshape((self.batch_size, self.seq_len, 112, 112, 3))
        #     batch_y = batch_y.reshape((self.batch_size, self.seq_len))
        #     batch_y = batch_y[:, -1]
        #     batch_y = batch_y.reshape((self.batch_size))
        # else:
        #     batch_img = batch_img.reshape((self.batch_size, 112, 112, 3))
        #     batch_flow = batch_flow.reshape((self.batch_size, 112, 112, 3))
        #     batch_y = batch_y.reshape((self.batch_size, self.seq_len, 1))
        #     batch_y = batch_y[:, -1]
        #     batch_y = batch_y.reshape((self.batch_size))

        # print(batch_x[0].shape)

        # batch_img = batch_img.reshape((self.batch_size, self.seq_len, 112, 112, 3))

        batch_x = np.array(self.signals[start_id:end_id])
        batch_x = batch_x.reshape((self.batch_size, self.seq_len, 1))

        batch_y = np.array(self.labels[start_id:end_id])
        batch_y = batch_y.reshape((self.batch_size, self.seq_len))

        return batch_x, batch_y

        # return [batch_img, batch_x], batch_y