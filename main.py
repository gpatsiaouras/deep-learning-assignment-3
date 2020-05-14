from os import path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os.path
import sys
import plotly.express as px
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def construct_dataset(data):
    input = []
    labels = []
    for idx in range(len(data) - window_size):
        input.append(data[idx: idx + window_size])
        # Label is the temperature of the last city
        labels.append(data[idx + window_size][-1][0])
    return np.array(input), np.array(labels)


def predict_recursively(predict_size):
    predictions = np.zeros((predict_size, 1))

    # Iterate from the length of the dataset up to plus the predict size
    for index in range(len(dataset) - predict_size, len(dataset)):
        # Get the prediction by inputting the last 50 values from the current index
        current_prediction = model.predict(dataset[np.newaxis, index - window_size:index])[0, 0]
        predictions[index - len(dataset) + predict_size] = current_prediction
        dataset[index, -1, temperature_index] = current_prediction

    return predictions


def plot_cities():
    plot_data = np.reshape(dataset, (len(dataset), 20))
    steps = 8736
    for feature in range(5):
        plt.clf()
        plt.plot(plot_data[:steps, 0 + feature], label="City 1")
        plt.plot(plot_data[:steps, 5 + feature], label="City 2")
        plt.plot(plot_data[:steps, 10 + feature], label="City 3")
        plt.plot(plot_data[:steps, 15 + feature], label="City 4")
        plt.legend()
        plt.title("Feature " + str(feature) + " one year")
        plt.xlabel("Time steps")
        plt.ylabel("Feature " + str(feature))
        plt.savefig("plots/data_vis_feature_" + str(feature) + "_" + str(steps) + "_" + ".png")
        plt.show()


if __name__ == "__main__":
    window_size = 50
    temperature_index = 2
    # Load data from file
    original_dataset = scipy.io.loadmat('data.mat')["X"].astype(float)
    dataset = original_dataset.copy()

    # Experiment with less features
    temperature_index = 0
    dataset = np.delete(dataset, np.s_[0, 1, 3, 4], axis=2) # Delete features
    dataset = np.delete(dataset, np.s_[0, 1, 2], axis=1) # Delete cities

    # Plot dataset per feature, all cities
    # plot_cities()

    # Scale whole dataset to range [0, 1] per feature
    scalers = []
    for feature in range(dataset.shape[2]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset[..., feature])
        dataset[..., feature] = scaler.transform(dataset[..., feature])
        scalers.append(scaler)
    # dataset = np.reshape(dataset, (len(dataset) * 4, 5))
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler.fit(dataset)
    # dataset = scaler.transform(dataset)
    # dataset = np.reshape(dataset, (len(dataset) // 4, 4, 5))

    train, test = construct_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=168, shuffle=False)

    n_steps = window_size
    n_cities = dataset.shape[1]
    n_features = dataset.shape[2]
    kernel_size = 1
    pool_size = 1
    epochs = 1

    model = None
    if path.exists('saved_models/saved_model_epochs_' + str(epochs)):
        model = load_model('saved_models/saved_model_epochs_' + str(epochs))
    else:
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(n_steps, n_cities, n_features)))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x=x_train, y=y_train, batch_size=50, epochs=epochs)
        # model.save('saved_models/saved_model_epochs_' + str(epochs))

    # Normal predict
    y_pred = model.predict(x=x_test, batch_size=50)
    # Recursive predict
    y_pred_rec = predict_recursively(predict_size=168)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_rec = mean_squared_error(y_test, y_pred_rec)
    mae_rec = mean_absolute_error(y_test, y_pred_rec)

    # Descale
    # temp = np.zeros((len(y_pred), 5, 1))
    # temp_rec = np.zeros((len(y_pred), 5, 1))
    # temp[:, 2] = y_pred
    # descaled_y_pred = scaler.inverse_transform(temp[:, :, 0])
    # temp_rec[:, 2] = y_pred_rec
    # descaled_y_pred_rec = scaler.inverse_transform(temp_rec[:, :, 0])
    # original_y_test = original_dataset[-168:, 3, 2]

    # Print predictions
    plt.clf()
    plt.plot(y_pred, label="Prediction")
    plt.plot(y_pred_rec, label="Prediction Recursively")
    plt.plot(y_test, label="Original")
    plt.legend()
    plt.title("Predictions")
    plt.xlabel("Time steps")
    plt.ylabel("Temperature")
    plt.savefig("predictions_both_rec_and_nor.png")
    plt.show()

    print("Normal prediction- Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse, mae))
    print("Recursive prediction - Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse_rec, mae_rec))
