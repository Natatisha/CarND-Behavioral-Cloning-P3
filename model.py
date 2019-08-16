import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import cv2
import os

from sklearn.model_selection import train_test_split

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Cropping2D
from keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE = (160, 320, 3)


def load_data(folder_path, file_name):
    # load data
    data = pd.read_csv(folder_path + file_name)

    # get center, left and right images
    images = pd.concat([data['center'], data['left'], data['right']])

    n = len(images)

    # allocate memory for loaded and flipped images (n * 2)
    X_train = np.empty(shape=((n, *IMG_SHAPE)), dtype=np.float32)

    # augment images and put all images into pre-allocated array
    for i, path in enumerate(images):
        image = mpimg.imread(folder_path + "IMG/" + os.path.basename(path))
        X_train[i] = image
    #         X_train[n + i] = cv2.flip(image, 1)

    # extract steering data
    steering_center = np.array(data['steering'], dtype=np.float32)

    # generate corrected steering data for left and right images
    correction = 0.23
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # put all measurements together into numpy array
    y_train = np.concatenate(
        (steering_center, steering_left, steering_right))

    return X_train, y_train


def build_network():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMG_SHAPE))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(kernel_size=5, filters=24, activation="relu"))
    model.add(Conv2D(kernel_size=5, filters=36, activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(kernel_size=5, filters=48, activation="relu"))
    model.add(Conv2D(kernel_size=3, filters=64, activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(kernel_size=3, filters=64, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss='mse')
    return model


if __name__ == '__main__':
    epochs = 5
    batch_size = 32

    X_train, y_train = load_data("./data/", "driving_log.csv")
    print(X_train.shape)
    print(y_train.shape)

    # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=True)

    model = build_network()
    model.summary()

    # steps_train = len(X_train) // batch_size
    # steps_valid = len(X_valid) // batch_size
    # train_datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
    #                                    horizontal_flip=True)
    #
    # train_datagen.fit(X_train)
    # train_datagen.standardize(X_valid)
    #
    # training_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    # model.fit_generator(training_generator, steps_per_epoch=steps_train,
    #                     validation_data=(X_valid, y_valid), validation_steps=steps_valid,
    #                     epochs=epochs)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, shuffle=True)
    model.save('model.h5')
