import numpy as np
import joblib
from neural_network import NeuralNetwork
from neural_network_config import parameters


def save_results(model, model_name):
    print('Serializing model...')
    with open(model_name, 'wb') as f:
        joblib.dump(model, f)


def get_saved_model(model_name='model.pkl'):
    print('Deserializing model...')
    with open(model_name, 'rb') as f:
        model = joblib.load(f)
    return model


def get_option():
    option = int(input('Enter 1 for training, 2 for prediction: '))
    if option == 1:
        return 'train'
    elif option == 2:
        return 'predict'
    return None


def get_train_data():
    train_data = []
    with open('segments.data', 'r') as f:
        for index, line in enumerate(f):
            if index is 0:
                continue
            values = line.split(',')
            values = [int(x) for x in values]
            x = np.array(values[:7]).reshape((7, 1))
            y = np.array(values[7:]).reshape((10, 1))
            train_data.append((x, y))
    return train_data


def train_network():
    print('Getting train data...')
    train_data = get_train_data()
    model = NeuralNetwork((7, 10, 10))
    print('Training network...')
    model.fit(training_data=train_data, **parameters)
    save_results(model, "model.pkl")


def predict_led():
    model = get_saved_model()
    x = input('Enter the digit values separated by space: ')
    x = [int(val) for val in x.split(' ')]
    x = np.array(x).reshape((7, 1))
    predictions = model.get_predictions([x])
    print(f'I think this is a {predictions[0]}')


def main():
    option = get_option()
    if option == 'train':
        train_network()
    elif option == 'predict':
        predict_led()


if __name__ == '__main__':
    main()
