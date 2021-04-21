#! /usr/bin/env python

import _pickle as cPickle, gzip
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
import sys
sys.path.append("..")
import utils
from utils import *
from train_utils import batchify_data, run_epoch, train_model


device = "cuda" if torch.cuda.is_available() else "cpu"

def run_network(hidden_layer_size=50, lr=0.1, momentum=0):
    # Load the dataset
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    # Split dataset into batches
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    ## Model specification TODO
    model = nn.Sequential(
              nn.Linear(784, hidden_layer_size),
              nn.ReLU(),
              nn.Linear(hidden_layer_size, 10),
            )

    model.to(device)
    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)

    ## Evaluate the model on test data
    loss, accuracy = run_epoch(test_batches, model.eval(), None)

    return loss, accuracy


def main():
    print(f"Using {device}.")
    LR = [10**i for i in range(-3, 2)]  # 5
    MOMENTUM = [i/10 for i in range(10)]  # 10
    HIDDEN_LAYER_SIZE = [i*10 for i in range (1, 10)]  # 9

    best_res = run_network()
    best_res = list(best_res)
    step = 0

    for lr in LR:
        step += 1
        print(f"Step: {step}")
        loss, accuracy = run_network(lr=lr)
        if accuracy > best_res[1]:
            best_res[0] = loss
            best_res[1] = accuracy
            best_lr = lr
        else:
            best_lr = .001
    
    for momentum in MOMENTUM:
        step += 1
        print(f"Step: {step}")
        loss, accuracy = run_network(momentum=momentum)
        if accuracy > best_res[1]:
            best_res[0] = loss
            best_res[1] = accuracy
            best_momentum = momentum
        else:
            best_momentum = 0
    
    for hidden_layer_size in HIDDEN_LAYER_SIZE:
        step += 1
        print(f"Step: {step}")
        loss, accuracy = run_network(hidden_layer_size)
        if accuracy > best_res[1]:
            best_res[0] = loss
            best_res[1] = accuracy
            best_hidden_layer_size = hidden_layer_size
        else:
            best_hidden_layer_size = 50

    best_res = run_network(best_hidden_layer_size, best_lr, best_momentum)

    print("Best result:")
    print (f"Loss: {best_res[0]:.4f} | Accuracy: {best_res[1]:.4f}\n")
    print("Best Parameters:")
    print (f"HLS: {best_hidden_layer_size} | LR: {best_lr} | MOM: {best_momentum}")


if __name__ == '__main__':
    np.random.seed(12321)  # for reproducibility
    torch.manual_seed(12321)  # for reproducibility
    main()
