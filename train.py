import sys
import os
import json
from pathlib import Path

import tensorflow as tf

from model import build_model
from losses import focal_loss
from data_sequence import DataSequence

# Check configuration file
args = sys.argv
if len(args) != 2:
    print("Usage: python train.py <path-to-config-file.json>")
    exit(0)

# Check and load config file
config_file = args[1]
if not os.path.isfile(config_file):
    print("Config is not a file: {}".format(config_file))
with open(config_file, "r") as infile:
    config = json.load(infile)

# Create experiment folder
experiment_folder = config["experiment_folder"]
Path(experiment_folder).mkdir(parents=True, exist_ok=True)

# Create data sequences
train_data = DataSequence(config["data"]["train_images"],
        config["data"]["train_labels"],
        batch_size=config["train_params"]["train_batchsize"],
        seq_len=config["model"]["seq_len"])
# TODO: data sequences for validation and testing


# Build model
model = build_model(seq_len=config["model"]["seq_len"])
if config["train_params"]["load_weights"]:
    model.load_weights(config["train_params"]["pretrained_weights"])

# Compile for training
opt = tf.keras.optimizers.Adam(lr=config["train_params"]["learning_rate"])
model.compile(loss=[focal_loss(alpha=.1, gamma=2)], optimizer=opt, metrics=["accuracy"])

# Fit
model_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment_folder, 'model.{epoch:03d}.h5')),
    tf.keras.callbacks.TensorBoard(log_dir=os.path.join(experiment_folder, 'logs')),
]
model.fit(train_data, epochs=config["train_params"]["n_epochs"], callbacks=model_callbacks, shuffle=True)