import os 
import pickle
import numpy as np

import torch

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(as_array=True):
    # Initialize variables
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Load all the paths of the pickle files
    cifar_path = "CIFAR-10"
    files_path = os.listdir(cifar_path)

    # Load training data
    for file in files_path: 
        filepath = os.path.join(cifar_path, file)
        if file.startswith("data_batch"):
            temp_dict = unpickle(filepath)
            X_train.extend(temp_dict[b'data'])
            y_train.extend(temp_dict[b'labels'])

    # Load testing data
    for file in files_path:
        filepath = os.path.join(cifar_path, file)
        if file.startswith("test_batch"):
            temp_dict = unpickle(filepath)
            X_test.append(temp_dict[b'data'])
            y_test.extend(temp_dict[b'labels'])

    if as_array:
        # Turn into numpy array 
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        # Reshape the data
        X_train = np.vstack(X_train)
        X_test= np.vstack(X_test)

    else: 
        # Reshape the data
        X_train = np.vstack(X_train).reshape(-1, 3, 32, 32) #-1 is the number of samples/images, 3 is the channnels, 32 is the height and 32 is the width
        X_test = np.vstack(X_test).reshape(-1, 3, 32, 32)

        # Turn into torch tensor
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)


    print('Data loaded succesfully!')
    return X_train, y_train, X_test, y_test