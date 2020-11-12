import requests
import json
import os
import numpy as np

def download_file(url, file_path):
    """Download from server"""
    myfile = requests.get(url)
    open(file_path, 'wb').write(myfile.content)


def read_label_file(label_file):

    with open(label_file, 'r') as infile:
        raw_data = json.load(infile)

    labels = raw_data["result"]["labels"]
    labels.sort(key=lambda x:x["_id"])
    videos = {}
    for label in labels:
        content = json.loads(label["content"])
        if content["label"] is not None:
            videos[label["video_id"]] = {}
            videos[label["video_id"]]["label"] = content["label"]

    return videos