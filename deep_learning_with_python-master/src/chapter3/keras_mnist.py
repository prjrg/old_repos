from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
input_size = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], input_size)
X_test = X_test.reshape(X_test.shape[0], input_size)

num_classes = 10
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

batch_size = 100
hidden_neurons = 100
epochs = 100

model = Sequential([
    Dense(hidden_neurons, input_dim=input_size),
    Activation('sigmoid'),
    Dense(num_classes),
    Activation('softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)

print('Test accuracy:', score[1])

weights = model.layers[0].get_weights()
fig = plt.figure()
w = weights[0].T
for neuron in range(hidden_neurons):
    ax = fig.add_subplot(10, 10, neuron + 1)
    ax.axis("off")
    ax.imshow(np.reshape(w[neuron], (28, 28)), cmap=cm.Greys_r)

plt.savefig("neuronimages.png", dpi=300)
plt.show()
