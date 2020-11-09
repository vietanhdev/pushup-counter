
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling1D
from tensorflow.keras.applications import MobileNetV2



def build_model(seq_len=10):

    model=Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same'), input_shape=(seq_len, 224, 224, 3)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Conv2D(32, (3, 3))))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dense(35)))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))

    return model