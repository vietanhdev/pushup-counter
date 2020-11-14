import json


input_path_pattern = "data/flow_{}.json"
def make_data(set_name):
    with open(input_path_pattern.format(set_name), "r") as infile:
        data = json.load(infile)
    images = []
    label_0 = 0
    label_1 = 0
    for video, frames in data.items():
        for frame_id, label in enumerate(frames):
            images.append({
                "img_name": "{}_{}".format(video, frame_id),
                "label": label
            })
            if label == 1:
                label_1 += 1
            else:
                label_0 += 1
    print("Set: ", set_name)
    print("label_1:", label_1)
    print("label_0:", label_0)
    return images

def write_label(videos, file_path):
    with open(file_path, 'w') as outfile:
        json.dump(videos, outfile)

write_label({"labels": make_data("train")}, "data/flow_images_train.json")
write_label({"labels": make_data("val")}, "data/flow_images_val.json")
write_label({"labels": make_data("test")}, "data/flow_images_test.json")

