import json
import numpy as np
from sklearn.model_selection import train_test_split
import keras.regularizers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt

JSON_PATH = "Valerio_data.json"


def load_data(json_path):
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    return data


def split_data(data):

    # Make numpy arrays from dictionary
    # 3D arrays (song, sample ,MFCC)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    # Split data (train, test, cross validation sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=123)

    input_shape = X_train[0].shape

    return input_shape, X_train, X_test, X_val, y_train, y_test, y_val


def build_model(input_shape):

    model = keras.Sequential()

    # Input Layer
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))

    # 1st hidden layer
    model.add(LSTM(128))
    model.add(Dropout(0.2))

    # 2nd hidden layer
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))

    # 3rd hidden layer
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

    # Save model
    model_name = 'Genre_model.h5'
    model.save(model_name)

    return model


def test_model(model, X_train, y_train, X_test, y_test, X_val, y_val):

    history = model.fit(X_train, y_train, epochs=50, batch_size=20, validation_data=(X_val, y_val), shuffle=False)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print('\nTest accuracy:', test_acc)

    return history


def plot_results(history):

    # Plot train and validation set accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Plot')
    plt.legend()

    # Plot train and validation set error
    plt.figure(1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error Plot')
    plt.legend()


if __name__ == "__main__":

    # Process data
    INPUT_SHAPE, X_TRAIN, X_TEST, X_VAL, Y_TRAIN, Y_TEST, Y_VAL = split_data(load_data(JSON_PATH))

    # Build model architecture
    MODEL = build_model(INPUT_SHAPE)

    # Check model accuracy
    HISTORY = test_model(MODEL, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, X_VAL, Y_VAL)

    # Visual representation of the results
    plot_results(HISTORY)
