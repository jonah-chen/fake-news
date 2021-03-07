import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import binary_crossentropy


from preprocessing import PADDING_CONST, generate_body

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


x_real, x_fake = generate_body(4096)

X = np.append(x_real, x_fake, axis=0)
Y = np.zeros((len(X),2), dtype=np.float32)

for i in range(len(x_real)):
    Y[i,0] = 1.0
for i in range(len(x_real), len(x_real)+len(x_fake)):
    Y[i,1] = 1.0


print(X.shape)
print(Y.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
print(Y_train)


model = Sequential()
model.add(Masking(mask_value=PADDING_CONST, input_shape=(None, 300,)))
model.add(LSTM(64))
model.add(Dense(2, activation="softmax"))

model.compile(metrics=['accuracy'], loss=binary_crossentropy, optimizer=SGD(lr=0.01, momentum=0.9))

model.summary()

model.fit(X_train, Y_train, batch_size=64, epochs=2, validation_data=(X_test,Y_test), use_multiprocessing=True)

model.save("model.h5")
print(model(x_real[:2], training=False))

# https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
