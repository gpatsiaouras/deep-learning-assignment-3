import numpy as np
import scipy.io
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


def construct_dataset(data, window_size):
    input = []
    labels = []
    for idx in range(len(data) - window_size):
        input.append(data[idx: idx + window_size])
        # Label is the temperature of the last city
        labels.append(data[idx + window_size][3][2])
    return np.array(input), np.array(labels)


if __name__ == "__main__":
    # Load data from file
    dataset = scipy.io.loadmat('data.mat')["X"].astype(float)

    # Scale whole dataset to range [0, 1]
    scalers = []
    for feature in range(dataset.shape[2]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset[..., feature])
        dataset[..., feature] = scaler.transform(dataset[..., feature])
        scalers.append(scaler)

    # Split dataset
    train_data = dataset[:-168]
    test_data = dataset[-168:]

    # Create windows based on the window size
    x_train, y_train = construct_dataset(train_data, 50)
    x_test, y_test = construct_dataset(test_data, 50)

    n_steps = 50
    n_cities = 4
    n_features = 5

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_cities, n_features)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(x=x_train, y=y_train, batch_size=1, epochs=10)

    y_pred = model.predict(x=x_test, batch_size=1)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print("Mean Squared Error: {}\nMean Absolute Error: {}".format(mse, mae))
