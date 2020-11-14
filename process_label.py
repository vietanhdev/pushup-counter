import json
import cv2
import os
import numpy as np
import random

INPUT_LABEL_FILE = "labels.json"
OUTPUT_LABEL_FILE = "data/labels-processed.json"
VIDEO_FOLDER = "/mnt/DATA/PUSHUP_PROJECT/processed"
HIT_PADDING = 5

with open(INPUT_LABEL_FILE, 'r') as infile:
    raw_data = json.load(infile)
    data = dict()

    labels = raw_data["result"]["labels"]
    labels.sort(key=lambda x:x["_id"])
    videos = {}
    for label in labels:
        content = json.loads(label["content"])
        if content["label"] is None or len(content["label"]) < 5:
            print(content["label"], label["video_id"])
            continue
        if content["label"] is not None:
            videos[label["video_id"]] = {}
            video_path = os.path.join(VIDEO_FOLDER, "{}.mp4".format(label["video_id"]))

            if not os.path.isfile(video_path):
                print("Not found:{}".format(video_path))

            video = cv2.VideoCapture(video_path)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

            
            videos[label["video_id"]]["n_frames"] = frame_count
            hit_frames = [False] * (frame_count + 1)
            
            l = np.array(content["label"])
            l1 = l[1:]
            l2 = l[:-1]
            avg_distance = np.abs(np.mean(l2 - l1))

            for l in content["label"]:
                pad = int(max(5, min(12, avg_distance // 4)))
                # print(pad)
                for i in range(max(0, l-0), min(frame_count-pad, l+pad)):
                    hit_frames[i] = True

            final_label = []
            for i in range(len(hit_frames)):
                if hit_frames[i]:
                    final_label.append(i)
            videos[label["video_id"]]["label"] = final_label

    # data["labels"] = videos
    # with open(OUTPUT_LABEL_FILE, 'w') as outfile:
    #     json.dump(data, outfile)


    def split_video_list(videos, ratio=0.8):
        video_keys = list(videos.keys())
        random.seed(42)
        random.shuffle(video_keys)
        train_keys = video_keys[:int(ratio*len(video_keys))]
        train_videos = {key: videos[key] for key in train_keys} 
        val_videos = {key: videos[key] for key in video_keys if key not in train_keys}
        return train_videos, val_videos

    train_videos, other_videos = split_video_list(videos, 0.7)
    val_videos, test_videos = split_video_list(other_videos, 0.5)
    
    def write_label(videos, file_path):
        data = {}
        data["labels"] = videos
        with open(file_path, 'w') as outfile:
            json.dump(data, outfile)

    write_label(train_videos, "data/train.json")
    write_label(val_videos, "data/val.json")
    write_label(test_videos, "data/test.json")
    write_label(videos, "data/all.json")

    print("Train:", len(train_videos))
    print("Val:", len(val_videos))
    print("Test:", len(test_videos))
    