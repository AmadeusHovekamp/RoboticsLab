from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf

# tf.logging.set_verbosity(tf.logging.DEBUG)


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    # print(test_set[0].shape, test_set[1].shape)
    # print(valid_set[0].shape, valid_set[1].shape)
    # print(train_set[0].shape, train_set[1].shape)
    # test_set = (test_set[0][:300], test_set[1][:300])
    # valid_set = (valid_set[0][:300], valid_set[1][:300])
    # train_set = (train_set[0][:300], train_set[1][:300])
    # print(test_set[0].shape, test_set[1].shape)
    # print(valid_set[0].shape, valid_set[1].shape)
    # print(train_set[0].shape, train_set[1].shape)

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')

    # return train_x, one_hot(train_y), valid_x, one_hot(valid_y),
    # test_x, one_hot(test_y)
    return train_x, train_y, valid_x, valid_y, test_x, test_y


def create_cnn_model(features, labels, mode):
    # Construct the convolutional neural network (CNN).

    # Input layer.
    layer_inputs = tf.reshape(features["x"], [-1, 28, 28, 1])

    # First convolutional layer.
    layer_conv_1 = tf.layers.conv2d(inputs=layer_inputs,
                                    filters=num_filters,
                                    kernel_size=filter_size,
                                    strides=1,
                                    padding='same',
                                    activation=tf.nn.relu)

    # First pooling layer.
    layer_pool_1 = tf.layers.max_pooling2d(inputs=layer_conv_1,
                                           pool_size=2,
                                           strides=1)

    # Second convolutional layer.
    layer_conv_2 = tf.layers.conv2d(inputs=layer_pool_1,
                                    filters=num_filters,
                                    kernel_size=filter_size,
                                    strides=1,
                                    padding='same',
                                    activation=tf.nn.relu)

    # Second pooling layer.
    layer_pool_2 = tf.layers.max_pooling2d(inputs=layer_conv_2,
                                           pool_size=2,
                                           strides=1)

    # Fully connected layer.
    layer_pool_2_flat = tf.contrib.layers.flatten(layer_pool_2)

    layer_fully_connected = tf.layers.dense(inputs=layer_pool_2_flat,
                                            units=128,
                                            activation=tf.nn.relu)

    # Output layer.
    layer_output_wo_softmax = tf.layers.dense(inputs=layer_fully_connected,
                                              units=10)

    predictions = {
        "classes": tf.argmax(input=layer_output_wo_softmax, axis=1),
        "probabilities": tf.nn.softmax(layer_output_wo_softmax,
                                       name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(
                labels=labels,
                logits=layer_output_wo_softmax)

    # Configure training.
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, train_op=train_op)

    # Evaluation metrics
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_and_validate(x_train,
                       y_train,
                       x_valid,
                       y_valid,
                       num_epochs,
                       lr,
                       num_filters,
                       batch_size,
                       filter_size):

    # Empty working directory
    model_dir = "./tmp/mnist_convnet_model"
    if os.path.exists(model_dir):
        for f in os.listdir(model_dir):
            fp = os.path.join(model_dir, f)
            try:
                if os.path.isfile(fp):
                    os.remove(fp)
                elif os.path.isdir(fp):
                    for f2 in os.listdir(fp):
                        fp2 = os.path.join(fp, f2)
                        os.remove(fp2)
            except Exception as e:
                print("ERROR", e)
                pass

    # Create the Estimator
    model = tf.estimator.Estimator(model_fn=create_cnn_model,
                                   model_dir=model_dir)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x_train},
                                                        y=y_train,
                                                        batch_size=batch_size,
                                                        num_epochs=1,
                                                        shuffle=True)
    learning_curve = []
    for e in range(num_epochs):
        print("\nStart training for epoch {0} ...".format(e+1))
        model.train(input_fn=train_input_fn)
        print("\nStart validating for epoch {0} ...".format(e+1))
        val_result = validate(x_valid, y_valid, model)
        print("\nValidation result for epoch {0}: {1}".format(e+1, val_result))
        learning_curve.append(val_result["accuracy"])

    return learning_curve, model


def validate(x_valid, y_valid, model):
    # Validates the network.

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_valid},
        y=y_valid,
        shuffle=False)
    eval_results = model.evaluate(input_fn=eval_input_fn)

    return eval_results


def test(x_test, y_test, model):
    # Test your network here by evaluating it on the test data.

    eval_results = validate(x_test, y_test, model)
    return 1 - eval_results["accuracy"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is \
                        not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=1e-3, type=float, nargs="?",
                        help="Learning rate for SGD")
    # Changed default value for num_filters from 32 to 16.
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters \
                        for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?",
                        help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be \
                        trained")
    parser.add_argument("--run_id", default=0, type=int, nargs="?",
                        help="Helps to identify different runs of an \
                        experiments")
    parser.add_argument("--filter_size", default=3, type=int, nargs="?",
                        help="Filter width and height")
    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filter_size = args.filter_size

    # train and test convolutional neural network
    x_train, y_train, x_valid, y_valid, x_test, y_test = mnist(args.input_path)

    learning_curve, model = train_and_validate(x_train, y_train,
                                               x_valid, y_valid,
                                               epochs, lr, num_filters,
                                               batch_size, filter_size)

    test_error = test(x_test, y_test, model)

    # print(test_error)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["filter_size"] = filter_size
    results["learning_curve"] = ["{}".format(i) for i in learning_curve]
    results["test_error"] = test_error

    # print("learning_curve", learning_curve)

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()
