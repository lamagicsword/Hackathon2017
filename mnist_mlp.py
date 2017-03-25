'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix
import numpy as np


def max_index(l: list):
    return max(
        range(len(l)), key=lambda i: l[i]
    )


def max_min(l: list):
    maximum = l[0]
    minimum = l[0]

    for item in l:
        if item > maximum:
            maximum = item
        elif item < minimum:
            minimum = item

    return maximum, minimum


def normalize(l: list):
    s = sum(l)

    return [
        float(item) / s for item in l
    ]


batch_size = 128
num_classes = 2
epochs = 1


train = np.loadtxt('dota2Train.csv', delimiter=',').transpose()
test = np.loadtxt('dota2Test.csv', delimiter=',').transpose()


# the data, shuffled and split between train and test sets
y_train = train[0]
x_train = train[4:].transpose()

y_test = test[0]
x_test = test[4:].transpose()

y_test = np.array([ 0 if i == -1 else i for i in y_test ])
y_train = np.array([ 0 if i == -1 else i for i in y_train ])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_train.shape, y_train.shape, 'train samples')
print(x_test.shape, y_test.shape, 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(1024, activation='relu', input_shape=(113,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='sigmoid'))
# # model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# # model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size, epochs=epochs,
                    verbose=0, validation_data=(x_test, y_test))

predicted = model.predict_proba(x_test, batch_size=batch_size, verbose=0)
actual = [max_index(y) for y in y_test]
predicted = [max_index(y) for y in predicted]


conf_matrix = confusion_matrix(actual, predicted)
conf_matrix = np.array([
    normalize(row) for row in conf_matrix
])

print(conf_matrix)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
