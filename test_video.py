from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
import cv2
import numpy as np

VIDEO = "/mnt/DATA/PUSHUP_PROJECT/processed/270.mp4"


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

model = load_model("model.43.h5", custom_objects={"focal_loss_fixed": focal_loss})

seq_len = 5

cap = cv2.VideoCapture(VIDEO)

while True:

    imgs = []
    for _ in range(seq_len):
        ret, img = cap.read()
        if not ret:
            exit(0)
        img = cv2.resize(img, (224, 224))
        img = img - 127.5
        img /= 127.5
        imgs.append(img)

    model_input = np.array(imgs)
    model_input = np.expand_dims(model_input, axis=0)

    pred = model.predict(model_input)

    for pr in pred[0]:
        print(pr)


    for i, im in enumerate(imgs):
        if pred[0][i] > 0.5:
            im = cv2.rectangle(im, (10, 10), (50, 50), (0, 0, 255), -1) 
        cv2.imshow("Result", im)
        cv2.waitKey(10)






