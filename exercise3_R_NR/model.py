import tensorflow as tf
import numpy as np

class Model:

    def __init__(self, learning_rate):
        self.lr = learning_rate
        self.num_filters = 16
        self.filter_size = 3

        self.x_dim = 96
        self.y_dim = 9

        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, self.x_dim, self.x_dim, 1])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, self.y_dim])

        # Define network
        #self.cnn_graph = self.create_cnn_graph(self.x_placeholder)
        #self.cnn_graph = self.create_improved_cnn_graph(self.x_placeholder)
        self.cnn_graph = self.create_improved_cnn_graph2(self.x_placeholder)

        # Loss and optimizer
        soft_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_placeholder, logits=self.cnn_graph)
        self.loss = tf.reduce_mean(soft_entropy)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        # Define measurements.
        self.y_prediction = tf.argmax(self.cnn_graph, 1)
        tf.add_to_collection('cnn_prediction', self.y_prediction)
        self.prediction_corrected = tf.equal(self.y_prediction, tf.argmax(self.y_placeholder, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.prediction_corrected, "float"))

        self.train_loss = np.array([])
        self.train_accuracy = np.array([])
        self.train_error = np.array([])
        self.valid_accuracy = np.array([])
        self.valid_error = np.array([])


        # self.prediction = tf.argmax(cnn_graph, 1)

        # TODO: Start tensorflow session
        # Using the context manager.
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def load(self, file_name):
        self.saver.restore(self.sess, file_name)


    def save(self, file_name):
        self.saver.save(self.sess, file_name)


    def create_cnn_graph(self, layer_inputs):
        layer_conv_1 = tf.layers.conv2d(inputs=layer_inputs,
                                        filters=self.num_filters,
                                        kernel_size=self.filter_size,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_1 = tf.layers.max_pooling2d(inputs=layer_conv_1,
                                               pool_size=2,
                                               strides=1)

        layer_conv_2 = tf.layers.conv2d(inputs=layer_pool_1,
                                        filters=self.num_filters,
                                        kernel_size=self.filter_size,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_2 = tf.layers.max_pooling2d(inputs=layer_conv_2,
                                               pool_size=2,
                                               strides=1)

        layer_pool_2_flat = tf.contrib.layers.flatten(layer_pool_2)

        layer_fully_connected = tf.layers.dense(inputs=layer_pool_2_flat,
                                                units=128,
                                                activation=tf.nn.relu)

        layer_output_wo_softmax = tf.layers.dense(inputs=layer_fully_connected,
                                              units=self.y_dim)

        return layer_output_wo_softmax


    def create_improved_cnn_graph(self, layer_inputs):
        layer_conv_1 = tf.layers.conv2d(inputs=layer_inputs,
                                        filters=16,
                                        kernel_size=7,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_1 = tf.layers.max_pooling2d(inputs=layer_conv_1,
                                               pool_size=2,
                                               strides=1)

        layer_conv_2 = tf.layers.conv2d(inputs=layer_pool_1,
                                        filters=16,
                                        kernel_size=7,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_2 = tf.layers.max_pooling2d(inputs=layer_conv_2,
                                               pool_size=2,
                                               strides=1)

        layer_pool_2_flat = tf.contrib.layers.flatten(layer_pool_2)

        layer_fully_connected = tf.layers.dense(inputs=layer_pool_2_flat,
                                                units=128,
                                                activation=tf.nn.relu)

        layer_output_wo_softmax = tf.layers.dense(inputs=layer_fully_connected,
                                              units=self.y_dim)

        return layer_output_wo_softmax



    def create_improved_cnn_graph2(self, layer_inputs):
        layer_conv_1 = tf.layers.conv2d(inputs=layer_inputs,
                                        filters=20,
                                        kernel_size=10,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_1 = tf.layers.max_pooling2d(inputs=layer_conv_1,
                                               pool_size=2,
                                               strides=1)

        layer_conv_2 = tf.layers.conv2d(inputs=layer_pool_1,
                                        filters=20,
                                        kernel_size=7,
                                        strides=1,
                                        padding='same',
                                        activation=tf.nn.relu)

        layer_pool_2 = tf.layers.max_pooling2d(inputs=layer_conv_2,
                                               pool_size=2,
                                               strides=1)

        layer_pool_2_flat = tf.contrib.layers.flatten(layer_pool_2)

        layer_fully_connected = tf.layers.dense(inputs=layer_pool_2_flat,
                                                units=128,
                                                activation=tf.nn.relu)

        layer_output_wo_softmax = tf.layers.dense(inputs=layer_fully_connected,
                                              units=self.y_dim)

        return layer_output_wo_softmax
