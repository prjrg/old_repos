from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import random
import numpy as np

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
input_size = X_train.shape[1] * X_train.shape[2] * 3
X_train = X_train.reshape(X_train.shape[0], input_size)
X_test = X_test.reshape(X_test.shape[0], input_size)

num_classes = 10
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

batch_size = 100
epochs = 100

model = Sequential([
    Dense(1024, input_dim=input_size),
    Activation('relu'),
    Dense(512),
    Activation('relu'),
    Dense(512),
    Activation('sigmoid'),
    Dense(num_classes),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, validation_data=(X_test, Y_test), verbose=1)
score = model.evaluate(X_test, Y_test, verbose=1)

print('Test accuracy:', score[1])

fig = plt.figure()
outer_grid = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
weights = model.layers[0].get_weights()

w = weights[0].T

for i, neuron in enumerate(random.sample(range(0, 1023), 100)):
    ax = plt.Subplot(fig, outer_grid[i])
    ax.imshow(np.mean(np.reshape(w[i], (32, 32, 3)), axis=2), cmap=cm.Greys_r)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
plt.show()

