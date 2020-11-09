
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2

def model():
    X_input = MobileNetV2(input_shape=(224, 244, 3), include_top=False, weights='imagenet')
    
    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(196, kernel_size=7, strides=4, padding="same")(X) # CONV1D
    X = BatchNormalization()(X) # Batch normalization
    X = Activation('relu')(X) # ReLu activation
    X = Dropout(0.5)(X) # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences = True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.5)(X) # dropout (use 0.8)
    X = BatchNormalization()(X) # Batch normalization
    
    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X) # GRU (use 128 units and return the sequences)
    X = Dropout(0.5)(X) # dropout (use 0.8)
    X = BatchNormalization()(X) # Batch normalization
    X = Dropout(0.5)(X) # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)
    model = Model(inputs = X_input, outputs = X)
    
    return model


model = model()
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

# model.fit(X, Y, batch_size = 5, epochs=1)
# loss, acc = model.evaluate(X_dev, Y_dev)