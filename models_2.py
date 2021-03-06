from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed, LSTM, Conv2D, GlobalAveragePooling2D, Flatten, Conv3D, ZeroPadding3D, MaxPooling3D, AveragePooling2D
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
import tensorflow as tf
import slowfast
from tensorflow.keras.layers import Input


def build_2_flow_model():

    input_img = Input(shape=(112, 112, 3))
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


def build_optical_flow_model():

    # input_img = Input(shape=(112, 112, 3))
    input_flow = Input(shape=(112, 112, 3))
    
    # x1 = MobileNetV2(weights='imagenet', include_top=False, alpha=0.5)(input_img)
    x2 = EfficientNetB0(weights='imagenet', include_top=False)(input_flow)

    # x1 = Flatten()(x1)
    # x1 = Dense(1024, activation='relu')(x1)
    # x1 = Dropout(0.5)(x1)

    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)

    # x = tf.keras.layers.Concatenate()([x1, x2])
    # x = Flatten()(x2)
    x = Dense(1, activation="sigmoid")(x2)

    model = Model(inputs=input_flow, outputs=x)

    return model

def lenet():
    model = tf.keras.Sequential()
    model.add(Conv2D(filters=2, kernel_size=(3, 3), activation='relu', input_shape=(112,112,3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(units=120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

