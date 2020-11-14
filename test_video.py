from tensorflow.keras.models import load_model
from losses import focal_loss
import cv2
import numpy as np
from models import build_c3d_model

VIDEO = "/mnt/DATA/PUSHUP_PROJECT/processed/270.mp4"

# model = build_c3d_model(16)
model = load_model("/mnt/DATA/PUSHUP_PROJECT/pushup-counter-slowfast/experiments/config-c3d-01/model.080.h5")
# model.load_weights("/mnt/DATA/PUSHUP_PROJECT/pushup-counter-slowfast/experiments/config-c3d-01/model.039.h5")

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






