from tensorflow.keras.models import load_model
from tensorflow. import Adam
from model import build_model
from losses import focal_loss
from data_sequence import DataSequence

model = load_model("model.020.h5", custom_objects={"focal_loss_fixed": focal_loss})

seq_len = 5

last_trigger_frame = time.time()
def update_count(count_var, current_value, seq):
    global last_trigger_frame
    if time.time() - last_trigger_frame < 0.5:
        return count_var, current_value
    for i in seq:
        if current_value == 0 and i == 1:
            count_var += 1
            last_trigger_frame = time.time()
            break
        current_value = i
    return count_var, current_value


import json
with open("data/labels-processed.json", "r") as infile:
    labels = json.load(infile)
videos = list(labels["labels"].keys())
videos = videos[10:]

for i in videos:
    video_path = "/mnt/DATA/PUSHUP_PROJECT/processed/{}.mp4".format(i)
    print(video_path)
    cap = cv2.VideoCapture(video_path)

    count = 0
    current_value = 0
    while True:

        imgs = []
        finished = False
        for _ in range(seq_len):
            ret, img = cap.read()
            if not ret:
                finished = True
                break
            img = cv2.resize(img, (224, 224))
            img = img - 127.5
            img /= 127.5
            imgs.append(img)

        if finished:
            break

        model_input = np.array(imgs)
        model_input = np.expand_dims(model_input, axis=0)

        pred = model.predict(model_input)
        pred[pred >= 0.5] = 1
        pred[pred < 1] = 0

        count, current_value = update_count(count, current_value, pred[0])

        for i, im in enumerate(imgs):
            if pred[0][i] == 1:
                im = cv2.rectangle(im, (10, 10), (50, 50), (0, 0, 255), -1) 
            im = cv2.putText(im, str(count), (100, 50), cv2.FONT_HERSHEY_SIMPLEX,  
                   2, (0, 255, 0), 3, cv2.LINE_AA) 
            
            im = cv2.resize(im, (500, 500))
            cv2.imshow("Result", im)
            cv2.waitKey(10)






