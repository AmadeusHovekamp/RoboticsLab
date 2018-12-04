from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import matplotlib.pyplot as plt
import random
import json

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')

    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)

    display_state(X_train[1000])
    # print("X_train.shape before rgb2gray:\n", X_train.shape)
    train_size = 1000
    valid_size = 30


    batches = len(X_train) // train_size

    X_train_temp = np.zeros(X_train.shape[:-1])
    X_valid_temp = np.zeros(X_valid.shape[:-1])
    for batch in range(batches):
        X_train_temp[batch*train_size : (batch+1)*train_size] = rgb2gray(X_train[batch*train_size : (batch+1)*train_size])
        X_valid_temp[batch*valid_size : (batch+1)*valid_size] = rgb2gray(X_valid[batch*valid_size : (batch+1)*valid_size])
    # print("X_train.shape after rgb2gray:\n", X_train.shape)


    X_train = X_train_temp.astype('float32').reshape(X_train.shape[0], 96, 96, 1)
    X_valid = X_valid_temp.astype('float32').reshape(X_valid.shape[0], 96, 96, 1)

    display_state(X_train[1000])

    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot()
    #    useful and you may want to return X_train_unhot ... as well.

    y_train_discretized = np.array([])
    y_train_action_indices = {}
    for i in range(9):
        y_train_action_indices[i] = np.array([]).astype('int')

    for i, y in enumerate(y_train):
        action_as_id = action_to_id(y)
        y_train_discretized = np.append(y_train_discretized, action_as_id)
        y_train_action_indices[action_as_id] = np.append(y_train_action_indices[action_as_id], i)

    #with open('action_distributions/data.json', 'w') as outfile:
    #    json.dump(y_train_action_indices, outfile)

    print("Distribution of actions: ")
    for a, indices in y_train_action_indices.items():
        print("\t{0}, {1}: {2}".format(a, get_action_name(a), len(indices)))

    y_train = one_hot(y_train_discretized, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    y_valid_discretized = np.array([])
    for y in y_valid:
        y_valid_discretized = np.append(y_valid_discretized, action_to_id(y))
    y_valid = one_hot(y_valid_discretized, classes=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    # plt.title("Histograms of train and validation set")
    # plt.hist(y_train_discretized, bins = 9)
    # plt.hist(y_valid_discretized, bins = 9)
    # plt.show()

    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).

    # TODO: add history
    # X_train_hist = np.array([X_train, X_train, X_train, X_train, X_train])

    return X_train, y_train, y_train_action_indices, X_valid, y_valid


def train_model(X_train, y_train, y_train_action_indices, X_valid, y_valid, n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    print("... train model")

    agent = Model(lr)

    tensorboard_eval = Evaluation(tensorboard_dir)

    # TODO: implement the training
    #
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    #
    # training loop
    train_loss = 0

    for batch_counter in range(n_minibatches):
        print(batch_counter)
        X_minibatch, y_minibatch = sample_minibatch(X_train, y_train, batch_size, y_train_action_indices, mode=1)

        # train
        _, tmp_loss = agent.sess.run([agent.train_op, agent.loss],
                                  feed_dict={agent.x_placeholder: X_minibatch,
                                             agent.y_placeholder: y_minibatch})


        agent.train_loss = np.append(agent.train_loss, tmp_loss)

        tmp_train_accuracy = agent.accuracy.eval({agent.x_placeholder: X_minibatch, agent.y_placeholder: y_minibatch}, session=agent.sess)
        agent.train_accuracy = np.append(agent.train_accuracy, tmp_train_accuracy)

        tmp_train_error = 1 - tmp_train_accuracy
        agent.train_error = np.append(agent.train_error, tmp_train_error)

        tmp_valid_accuracy = agent.accuracy.eval({agent.x_placeholder: X_valid, agent.y_placeholder: y_valid}, session=agent.sess)
        agent.valid_accuracy = np.append(agent.valid_accuracy, tmp_valid_accuracy)

        tmp_valid_error = 1 - tmp_valid_accuracy
        agent.valid_error = np.append(agent.valid_error, tmp_valid_error)

        # if batch_counter % 10 == 0:
        tensorboard_eval.write_episode_data(batch_counter, {"loss": tmp_loss,
                                                "train_accuracy": tmp_train_accuracy,
                                                "train_error": tmp_train_error,
                                                "valid_accuracy": tmp_valid_accuracy,
                                                "valid_error": tmp_valid_error})

    # TODO: save your agent
    model_file = os.path.join(model_dir, "agent.ckpt")
    agent.save(model_file)
    print("Model saved in file: %s" % model_file)

    if not os.path.exists("results_measurements"):
        os.mkdir("results_measurements")

    fname = os.path.join("results_measurements", "train_loss.json")
    fh = open(fname, "w")
    json.dump(["{}".format(i) for i in agent.train_loss], fh)
    fh.close()

    fname = os.path.join("results_measurements", "train_accuracy.json")
    fh = open(fname, "w")
    json.dump(["{}".format(i) for i in agent.train_accuracy], fh)
    fh.close()

    fname = os.path.join("results_measurements", "train_error.json")
    fh = open(fname, "w")
    json.dump(["{}".format(i) for i in agent.train_error], fh)
    fh.close()

    fname = os.path.join("results_measurements", "valid_accuracy.json")
    fh = open(fname, "w")
    json.dump(["{}".format(i) for i in agent.valid_accuracy], fh)
    fh.close()

    fname = os.path.join("results_measurements", "valid_error.json")
    fh = open(fname, "w")
    json.dump(["{}".format(i) for i in agent.valid_error], fh)
    fh.close()
    # print("Results:")
    # print("agent.train_accuracy:", agent.train_accuracy)


def sample_minibatch(X, y, batch_size, y_train_action_indices, mode=0):
    print("sample_minibatch called!")
    if mode == 0:
        indices = range(len(X))
        indices = random.sample(indices, batch_size)
        X_minibatch = X[indices]
        y_minibatch = y[indices]
    else:
        tmp = np.array([]).astype('int')
        for b in range(batch_size):
            actions = range(5)#9)
            action = random.sample(actions, 1)[0]
            indices = range(len(y_train_action_indices[action]))
            index = random.sample(indices, 1)[0]
            tmp = np.append(tmp, y_train_action_indices[action][index])
        X_minibatch = X[tmp]
        y_minibatch = y[tmp]
    return X_minibatch, y_minibatch

if __name__ == "__main__":

    # read data
    X_train, y_train, X_valid, y_valid = read_data("./data", 0.03)

    # preprocess data
    X_train, y_train, y_train_action_indices, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, y_train_action_indices, X_valid, y_valid, n_minibatches=1000, batch_size=64, lr=0.0001)
