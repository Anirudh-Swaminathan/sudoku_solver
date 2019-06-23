#! /bin/env/python

"""This is a script to train a CNN to recognize handwritten digits from MNIST"""
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

import numpy as np
import np_utils


def prepare_data(trX, trY, teX, teY):
    """Function to prepare the data"""
    # reshape to be [samples][pixels][width][height]
    X_train = trX.reshape(trX.shape[0], 28, 28, 1).astype('float32')
    X_test = teX.reshape(teX.shape[0], 28, 28, 1).astype('float32')

    # normalize inputs from 0-255 to 0-1
    X_train = X_train / 255
    X_test = X_test / 255

    # one hot encode outputs
    y_train = keras.utils.to_categorical(trY)
    y_test = keras.utils.to_categorical(teY)
    num_classes = y_test.shape[1]
    return X_train, y_train, X_test, y_test


def conv_model():
    """Function to define the architecture of the CNN for MNIST classification"""
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=(28, 28, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def save_model(model):
    model_json = model.to_json()
    open('mnist.json', 'w').write(model_json)
    model.save_weights('mnist_weights.h5', overwrite=True)


def main():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    seed = 7
    np.random.seed(seed)

    tr_X, tr_y, te_X, te_y = prepare_data(X_train, y_train, X_test, y_test)
    model = conv_model()
    model.fit(tr_X, tr_y, validation_data=(te_X, te_y), epochs=10, batch_size=200, verbose=1)

    # save the trained model
    save_model(model)

    scores = model.evaluate(te_X, te_y, verbose=1)
    print scores
    model.save("mnist.h5")


if __name__ == "__main__":
    main()
