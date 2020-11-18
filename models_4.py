from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Conv2D, GlobalAveragePooling2D, Flatten, Conv3D, ZeroPadding3D, MaxPooling3D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import tensorflow as tf
import slowfast
from tensorflow.keras.layers import Input


def build_lstm_model(seq_len):

    model = Sequential([
        Input(shape=(seq_len, 1)),
        tf.keras.layers.LSTM(16, activation=tf.nn.relu,
                        return_sequences=True),    
        tf.keras.layers.LSTM(16, activation=tf.nn.relu,
                        return_sequences=True),  
        Dense(1, activation="sigmoid")
    ])

    return model


def build_2flow_cnn_lstm(seq_len):
    input_img = Input(shape=(seq_len, 112, 112, 3))
    input_flow = Input(shape=(112, 112, 3))
    
    x1 = MobileNetV2(weights='imagenet', include_top=False, alpha=0.5)(input_img)
    x2 = EfficientNetB0(weights='imagenet', include_top=False)(input_flow)

    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)

    x = tf.keras.layers.Concatenate()([x1, x2])
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[input_img, input_flow], outputs=x)

    return model