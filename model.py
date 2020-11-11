from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Conv2D, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def build_model(seq_len=10):

    model = Sequential()    
    model.add(TimeDistributed(MobileNetV2(weights='imagenet', include_top=False), input_shape=(seq_len, 224, 224, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(35)))
    model.add(LSTM(128, return_sequences=True))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))

    return model