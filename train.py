from tensorflow.keras.optimizers import Adam
from model import build_model
from data_sequence import DataSequence
import tensorflow as tf
from tensorflow.keras import backend as K

data = DataSequence("/mnt/DATA/PUSHUP_PROJECT/images", "data/labels-processed.json", batch_size=8, seq_len=5)

model = build_model(seq_len=5)

model.load_weights("model.40.h5")


def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss=[focal_loss(alpha=.1, gamma=2)], optimizer=opt, metrics=["accuracy"])


my_callbacks = [
    # tf.keras.callbacks.EarlyStopping(patience=3),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]
model.fit(data, epochs=200, callbacks=my_callbacks, shuffle=True)

loss, acc = model.evaluate(data)