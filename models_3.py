from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Conv2D, GlobalAveragePooling2D, Flatten, Conv3D, ZeroPadding3D, MaxPooling3D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import tensorflow as tf
import slowfast
from tensorflow.keras.layers import Input


def build_2_flow_model(seq_len):

    input_img = Input(shape=(seq_len, 112, 112, 3))
    input_flow = Input(shape=(seq_len, 112, 112, 3))
    
    x1 = Conv3D(64, (1, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(input_img)
    x1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), padding="valid")(x1)
    x1 = Conv3D(128, (3, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(x1)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x1)
    x1 = Conv3D(256, (3, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(x1)
    x1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x1)

    x2 = Conv3D(64, (1, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(input_flow)
    x2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(2, 2, 2), padding="valid")(x2)
    x2 = Conv3D(128, (3, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(x2)
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x2)
    x2 = Conv3D(256, (3, 3, 3), activation="relu", strides=(1, 1, 1), padding="same")(x2)
    x2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid")(x2)

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
