from tensorflow.keras.optimizers import Adam
from model import build_model
from data_sequence import DataSequence


data = DataSequence("/mnt/DATA/PUSHUP_PROJECT/images", "data/labels-processed.json", batch_size=8, seq_len=10)

model = build_model(seq_len=10)

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(data, epochs=10)

loss, acc = model.evaluate(data)