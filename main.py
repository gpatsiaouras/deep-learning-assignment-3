import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def construct_dataset(data):
    input = []
    labels = []
    for idx in range(len(data) - window_size):
        input.append(data[idx: idx + window_size])
        # Label is the temperature of the last city
        labels.append(data[idx + window_size][3][2])
    return np.array(input), np.array(labels)


def predict_recursively(predict_size):
    predictions = np.zeros((predict_size, 1))

    # Iterate from the length of the dataset up to plus the predict size
    for index in range(len(dataset) - predict_size, len(dataset)):
        # Get the prediction by inputting the last 50 values from the current index
        current_prediction = model.predict(dataset[np.newaxis, index - window_size:index])
        predictions[index - len(dataset) + predict_size] = current_prediction
        dataset[index, 3, 2] = current_prediction

    return predictions


if __name__ == "__main__":
    window_size = 500
    # Load data from file
    original_dataset = scipy.io.loadmat('data.mat')["X"].astype(float)
    dataset = original_dataset.copy()

    # Scale whole dataset to range [0, 1] per feature
    scalers = []
    for feature in range(dataset.shape[2]):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(dataset[..., feature])
        dataset[..., feature] = scaler.transform(dataset[..., feature])
        scalers.append(scaler)

    train, test = construct_dataset(dataset)
    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=168, shuffle=False)

    n_steps = window_size
    n_cities = 4
    n_features = 5

    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_cities, n_features)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(x=x_train, y=y_train, batch_size=50, epochs=20)

    # Normal predict
    y_pred = model.predict(x=x_test, batch_size=50)
    # Recursive predict
    y_pred_rec = predict_recursively(predict_size=168)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse_rec = mean_squared_error(y_test, y_pred_rec)
    mae_rec = mean_absolute_error(y_test, y_pred_rec)

    # Descale
    temp = np.zeros((len(y_pred), 4, 1))
    temp[:, 3] = y_pred
    temp[:, 2] = y_pred_rec
    descaled_y_pred = scalers[2].inverse_transform(temp[:, :, 0])
    original_y_test = original_dataset[-168:, 3, 2]

    plt.plot(descaled_y_pred[:, 3], label="Prediction")
    plt.plot(descaled_y_pred[:, 2], label="Prediction Recursively")
    plt.plot(original_y_test, label="Original")
    plt.legend()
    plt.show()

    print("Normal prediction- Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse, mae))
    print("Recursive prediction - Mean Squared Error: {}\nMean Absolute Error: {}\n\n".format(mse_rec, mae_rec))
