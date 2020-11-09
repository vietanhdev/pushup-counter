from tensorflow.keras.optimizers import Adam
from model import build_model
from data_sequence import DataSequence
import tensorflow as tf

data = DataSequence("/mnt/DATA/PUSHUP_PROJECT/images", "data/labels-processed.json", batch_size=8, seq_len=10)

model = build_model(seq_len=10)

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(data, epochs=40, callbacks=my_callbacks)

loss, acc = model.evaluate(data)