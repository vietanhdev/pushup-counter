import json
import numpy as np
import random

with open("data/pushing_labels.json", "r") as infile: 
    flow_data = json.load(infile)

with open("data/all.json") as infile:
    trigger_data = json.load(infile)["labels"]


def is_pushing_up(begin, end, video):
    for v in video["label"]:
        if v >= begin and v <= end:
            return True
    return False

new_flow_data = {}
for key, data in flow_data.items():
    flow = data["bin_label"]
    signal = data["raw_signal"]
    new_flow = [0] * len(flow)
    flow.append(0)
    prev = 0
    begin = 0
    end = 0
    # print(trigger_data)
    if key not in trigger_data:
        continue
    video = trigger_data[key]
    for i, frame in enumerate(flow):
        if flow[i] == 1 and prev == 0:
            begin = i
        elif flow[i] == 0 and prev == 1:
            end = i - 1
            if is_pushing_up(begin, end, video):
                for j in range(max(begin, end-10), end+1):
                    new_flow[j] = 1
        prev = flow[i]
    new_flow_data[key] = {}
    new_flow_data[key]["label"] = new_flow
    new_flow_data[key]["signal"] = signal


def split_video_list(videos, ratio=0.8):
    video_keys = list(videos.keys())
    random.seed(42)
    random.shuffle(video_keys)
    train_keys = video_keys[:int(ratio*len(video_keys))]
    train_videos = {key: videos[key] for key in train_keys} 
    val_videos = {key: videos[key] for key in video_keys if key not in train_keys}
    return train_videos, val_videos

def write_label(videos, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(videos, outfile)

train_videos, other_videos = split_video_list(new_flow_data, 0.7)
val_videos, test_videos = split_video_list(other_videos, 0.5)
write_label(train_videos, "data/flow_train.json")
write_label(val_videos, "data/flow_val.json")
write_label(test_videos, "data/flow_test.json")
write_label(new_flow_data, "data/flow_all.json")

print("Train:", len(train_videos))
print("Val:", len(val_videos))
print("Test:", len(test_videos))

