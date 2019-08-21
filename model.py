import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Lambda, Cropping2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers

IMG_SHAPE = (160, 320, 3)


def load_samples(folder_path, file_name):
    # load data
    data = pd.read_csv(folder_path + file_name)
    # we want to use right and left images, so let's stack all the data into one array
    images = pd.concat([data['center'], data['left'], data['right']])

    # get steering values
    steering_center = data['steering']
    # tuned correction angle for right and left images
    correction = 0.24
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    # stack all measurements into one array
    measurements = pd.concat([steering_center, steering_left, steering_right])

    print("Loaded {} samples".format(len(images)))

    # as result we have 2D array with two columns: images, measurements
    return pd.concat([images, measurements], axis=1)


def generator(samples, batch_size=32, folder_path="./data/"):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples.iloc[offset:offset + batch_size]

            images = []
            angles = []
            for i, batch_sample in batch_samples.iterrows():
                name = (folder_path + "IMG/" + os.path.basename(batch_sample[0]))
                # read image from the path
                image = mpimg.imread(name)
                angle = float(batch_sample['steering'])
                # divide data into X and y values, where angles = labels, images = features
                images.append(image)
                angles.append(angle)

            X_train = np.array(images, dtype=np.float32)
            y_train = np.array(angles, dtype=np.float32)
            yield shuffle(X_train, y_train)


def build_network():
    beta = 0.01
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=IMG_SHAPE))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Conv2D(kernel_size=5, filters=24, activation="relu", kernel_regularizer=regularizers.l2(beta)))
    model.add(Conv2D(kernel_size=5, filters=36, activation="relu", kernel_regularizer=regularizers.l2(beta)))
    model.add(MaxPool2D())

    model.add(Conv2D(kernel_size=5, filters=48, activation="relu", kernel_regularizer=regularizers.l2(beta)))
    model.add(Conv2D(kernel_size=3, filters=64, activation="relu", kernel_regularizer=regularizers.l2(beta)))
    model.add(MaxPool2D())

    model.add(Conv2D(kernel_size=3, filters=64, activation="relu", kernel_regularizer=regularizers.l2(beta)))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))

    model.compile(optimizer=Adam(), loss='mse')
    return model


def plot_learning_curve(loss, validation_loss):
    plt.plot(loss)
    plt.plot(validation_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    epochs = 8
    batch_size = 32

    samples = load_samples("./data/", "driving_log.csv")
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print(train_samples.shape)
    print(validation_samples.shape)

    train_generator = generator(train_samples, batch_size=batch_size)
    validation_generator = generator(validation_samples, batch_size=batch_size)

    model = build_network()
    model.summary()

    steps_train = len(train_samples) // batch_size
    steps_valid = len(validation_samples) // batch_size

    checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

    history = model.fit_generator(train_generator, steps_per_epoch=steps_train,
                                  validation_data=validation_generator, validation_steps=steps_valid,
                                  epochs=epochs, callbacks=[checkpoint, es])

    plot_learning_curve(history.history['loss'], history.history['val_loss'])
