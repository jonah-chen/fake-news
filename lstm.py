import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Masking, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from time import time

tensorboard = TensorBoard(log_dir=f"LOGS/{time()}", histogram_freq=1, write_images=True)
checkpoint = ModelCheckpoint("checkpoint/ckpt.h5", save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max')


from preprocessing import PADDING_CONST, generate_body, generate_title, get_dataset_2, get_dataset_3



# TODO: 
# 1. Process the data for training.
#       a. combine the real and fake
#       b. generate the "y" or the labels
#       c. shuffle the data, split into validation/train sets
#
#
# 2. Build the model and train on a smaller set
# 
# 3. Tune hyperparemeters (if there's time)
#
# 4. Train model on full set and test (overnight)


x_real, x_fake = generate_body(4000) # get the first 4000 real and fake articles from this dataset


X = np.append(x_real, x_fake, axis=0)

Y = np.zeros((len(X),2), dtype=np.float32)



for i in range(len(x_real)):
    Y[i,0] = 1.0
for i in range(len(x_real), len(x_real)+len(x_fake)):
    Y[i,1] = 1.0

del x_real
del x_fake

_X, _Y = get_dataset_2() # extract all the articles from the 2nd dataset
X = np.append(X, _X, axis=0)
Y = np.append(Y, _Y, axis=0)

_X, _Y = get_dataset_3() # extract all the articles from the 3rd dataset
X = np.append(X, _X, axis=0)
Y = np.append(Y, _Y, axis=0)

del _X
del _Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

del X
del Y

# Build the simple model

model = Sequential()
model.add(Masking(mask_value=PADDING_CONST, input_shape=(None, 300,)))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dropout(0.3))
model.add(Dense(2, activation="softmax"))

model.compile(metrics=['accuracy'], loss=binary_crossentropy, optimizer=SGD(lr=0.01, momentum=0.9))

model.summary()

model.fit(X_train, Y_train, batch_size=32, epochs=20, validation_data=(X_test,Y_test), use_multiprocessing=True, callbacks=[tensorboard, checkpoint])

model.load_weights("checkpoint/ckpt.h5")
model.save("model_final.h5")
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
