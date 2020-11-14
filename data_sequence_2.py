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

    def __init__(self, images_folder, flow_folder, label_file, batch_size):

        self.batch_size = batch_size
        self.images_folder = images_folder
        self.flow_folder = flow_folder
        with open(label_file, "r") as infile:
            labels = json.load(infile)["labels"]

        self.image_names = []
        self.labels = []
        for l in labels:
            image_name = l["img_name"]
            flow_path =  os.path.join(self.flow_folder, image_name + ".png")
            img_path =  os.path.join(self.images_folder, image_name + ".png")
            if os.path.isfile(flow_path) and os.path.isfile(img_path):
                self.image_names.append(l["img_name"])
                self.labels.append(l["label"])

        # self.image_names =  self.image_names[:100]
        # self.labels =  self.labels[:100]

        self.n_samples = len(self.image_names)
        self.n_batches = math.floor(self.n_samples / self.batch_size)


    def __len__(self):
        return self.n_batches

    def __getitem__(self, idx):

        start_id = idx * self.batch_size
        end_id = (idx + 1) * self.batch_size
        batch_image_names = self.image_names[start_id:end_id] 
    
        batch_img = []
        for image_name in batch_image_names:
            img_path =  os.path.join(self.images_folder, image_name + ".png")
            img = cv2.imread(img_path)
            if img is None:
                print(img_path)
                exit(1)
            batch_img.append(img)
        batch_img = np.array(batch_img)


        ia.seed(1)
        seq = iaa.Sequential([
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2)
        ], random_order=True) # apply augmenters in random order
        batch_img = seq(images=batch_img)

        batch_img = batch_img - 127.5
        batch_img /=  127.5

        # for img in batch_img:
        #     cv2.imshow("img", img)
        #     cv2.waitKey(0)


        batch_flow = []
        for image_name in batch_image_names:
            flow_path =  os.path.join(self.flow_folder, image_name + ".png")
            flow_img = cv2.imread(flow_path)
            if flow_img is None:
                print(flow_path)
                exit(1)
            batch_flow.append(flow_img)
        batch_flow = np.array(batch_flow)

        batch_x = [batch_img, batch_flow]
        batch_y = np.array(self.labels[start_id:end_id])

        # print(batch_x[0].shape)

        return batch_x, batch_y