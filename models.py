from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Conv2D, GlobalAveragePooling2D, Flatten, Conv3D, ZeroPadding3D, MaxPooling3D
from tensorflow.keras.applications import MobileNetV2
import tensorflow as tf
import slowfast
from tensorflow.keras.layers import Input


def build_cnn_lstm_model(seq_len=10):
    model = Sequential()    
    model.add(TimeDistributed(, input_shape=(seq_len, 112, 112, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(35)))
    model.add(LSTM(10, return_sequences=True))
    Dense(1, activation = "sigmoid")
    return model


def build_cnn_lstm_model2(seq_len=10):
    model = Sequential()    
    model.add(TimeDistributed(MobileNetV2(weights='imagenet', include_top=False, alpha=0.35), input_shape=(seq_len, 224, 224, 3)))
    model.add(TimeDistributed(GlobalAveragePooling2D()))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(512)))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(35)))
    model.add(LSTM(10, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))
    return model

def build_c3d_model(seq_len=10):

    # https://github.com/karolzak/conv3d-video-action-recognition/blob/master/python/c3dmodel.py

    model = Sequential()

    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation="relu",name="conv1", 
                     input_shape=(seq_len, 112, 112, 3),
                     strides=(1, 1, 1), padding="same"))  
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), name="pool1", padding="valid"))

    # 2nd layer group  
    model.add(Conv3D(128, (3, 3, 3), activation="relu",name="conv2", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool2", padding="valid"))

    # 3rd layer group   
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3a", 
                     strides=(1, 1, 1), padding="same"))
    model.add(Conv3D(256, (3, 3, 3), activation="relu",name="conv3b", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool3", padding="valid"))

    # 4th layer group  
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4a", 
                     strides=(1, 1, 1), padding="same"))   
    model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv4b", 
                     strides=(1, 1, 1), padding="same"))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool4", padding="valid"))
    model.add(Flatten())

    model.add(LSTM(32, return_sequences=True))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(1, activation = "sigmoid")))

    # 5th layer group  
    # model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5a", 
    #                  strides=(1, 1, 1), padding="same"))   
    # model.add(Conv3D(512, (3, 3, 3), activation="relu",name="conv5b",
    #                   strides=(1, 1, 1), padding="same"))
    # model.add(ZeroPadding3D(padding=(0, 1, 1)))	
    # model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), name="pool5", padding="valid"))
    # model.add(Flatten())

    # # FC layers group
    # model.add(Dense(1024, activation='relu', name='fc6'))
    # model.add(Dropout(.5))
    # model.add(Dense(512, activation='relu', name='fc7'))
    # model.add(Dropout(.5))
    # model.add(Dense(seq_len, activation='sigmoid', name='fc8'))

    return model


def slowfast_resnet50(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 6, 3], slowfast.bottleneck, **kwargs)
    return model
def slowfast_resnet101(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 23, 3], slowfast.bottleneck, **kwargs)
    return model
def slowfast_resnet152(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 8, 36, 3], slowfast.bottleneck, **kwargs)
    return model
def slowfast_resnet200(inputs, **kwargs):
    model = slowfast.Slow_body(inputs, [3, 24, 36, 3], slowfast.bottleneck, **kwargs)
    return model
def build_slowfast_model():
    # x = tf.random_uniform([4, 64, 224, 224, 3])
    inputs = Input(shape=(None, None, None, 3))
    model = slowfast_resnet50(inputs, num_classes=32)
    return model
