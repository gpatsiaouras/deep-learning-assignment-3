from os import path
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
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
    dataset_copy = dataset.copy()
    predictions = np.zeros((predict_size, 1))

    # Iterate from the length of the dataset up to plus the predict size
    for index in range(len(dataset_copy) - predict_size, len(dataset_copy)):
        # Get the prediction by inputting the last 50 values from the current index
        current_prediction = model.predict(dataset_copy[np.newaxis, index - window_size:index])[0, 0]
        predictions[index - len(dataset_copy) + predict_size] = current_prediction
        dataset_copy[index, -1, temperature_index] = current_prediction

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
    temperature_index = 2 # default index of temperature is 2
    # Load data from file
    original_dataset = scipy.io.loadmat('data.mat')["X"].astype(float)
    dataset = original_dataset.copy()

    # Experiment with less features
    name_of_graph = "Predictions (ws=50, city 3, feature [0,1,2], epochs 50"
    filename_of_graph = "plots/prediction_ws50_city3_feature0-1-2_ep50.png"
    model_filename = "prediction_ws50_city3_feature0-1-2_ep50"
    epochs = 20
    temperature_index = 2 # Indicate the index where the temperature is located in the dataset
    dataset = dataset[:, 3:, [0, 1, 2]] # keep only the last city and feature 0,1,2

    # Plot dataset per feature, all cities. Uncomment to print original data per city per feature
    # plot_cities()

    # Scale whole dataset to range [0, 1] per feature, and save the scales to be able to scale back
    scalers = []
    for feature in range(dataset.shape[2]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset[..., feature])
        dataset[..., feature] = scaler.transform(dataset[..., feature])
        scalers.append(scaler)

    # Construct a dataset of batches according to the window_size
    train, test = construct_dataset(dataset)
    # Split the dataset to prepare for training
    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=168, shuffle=False)

    # Dynamic declaration of the input_shape to allow training with different number of cities and features
    n_steps = window_size
    n_cities = dataset.shape[1]
    n_features = dataset.shape[2]
    kernel_size = 2
    pool_size = 2

    model = None
    # Load the model from the file if there is one from previous training
    if path.exists('saved_models/' + model_filename):
        model = load_model('saved_models/' + model_filename)
    else:
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=kernel_size, activation='relu', input_shape=(n_steps, n_cities, n_features)))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(x=x_train, y=y_train, batch_size=50, epochs=epochs)
        model.save('saved_models/' + model_filename)

    # Normal predict
    y_pred = model.predict(x=x_test, batch_size=50)
    # Recursive predict
    y_pred_rec = predict_recursively(predict_size=168)

    # Calculate metrics for both recursive and normal predict
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_rec = mean_squared_error(y_test, y_pred_rec)
    mae_rec = mean_absolute_error(y_test, y_pred_rec)

    # Descale
    # Scaler needs the same shape to be able to inverse the transformation. We just use an zero array with the
    # appropriate shape and only use the feature 2 in the end
    temp = np.zeros((len(y_pred), dataset.shape[1]))
    temp_rec = np.zeros((len(y_pred), dataset.shape[1]))
    temp[:, 2:] = y_pred
    temp_rec[:, 2:] = y_pred_rec
    original_y_test = scalers[2].inverse_transform(dataset[-168:, :, 2])
    descaled_y_pred = scalers[2].inverse_transform(temp)
    descaled_y_pred_rec = scalers[2].inverse_transform(temp_rec)

    # Print predictions
    plt.clf()
    # Only the 2 index column contains the data.
    plt.plot(descaled_y_pred[:, 2], label="Prediction")
    plt.plot(descaled_y_pred_rec[:, 2], label="Prediction Recursively")
    plt.plot(original_y_test[:, 2], label="Original")
    plt.legend()
    plt.title(name_of_graph)
    plt.xlabel("Time steps")
    plt.ylabel("Temperature")
    plt.savefig(filename_of_graph)
    plt.show()

    # Print metrics
    print("Normal prediction- Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse, mae))
    print("Recursive prediction - Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse_rec, mae_rec))
