import tensorflow as tf

class Evaluation:

    def __init__(self, store_dir):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(store_dir)

        self.tf_loss = tf.placeholder(tf.float32, name="loss_summary")
        tf.summary.scalar("loss", self.tf_loss)

        # TODO: define more metrics you want to plot during training (e.g. training/validation accuracy)
        self.tf_train_accuracy = tf.placeholder(tf.float32, name="train_accuracy_summary")
        tf.summary.scalar("train_accuracy", self.tf_train_accuracy)

        self.tf_train_error = tf.placeholder(tf.float32, name="train_error_summary")
        tf.summary.scalar("train_error", self.tf_train_error)

        self.tf_valid_accuracy = tf.placeholder(tf.float32, name="valid_accuracy_summary")
        tf.summary.scalar("valid_accuracy", self.tf_valid_accuracy)

        self.tf_valid_error = tf.placeholder(tf.float32, name="valid_error_summary")
        tf.summary.scalar("valid_error", self.tf_valid_error)

        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):

       # TODO: add more metrics to the summary
       summary = self.sess.run(self.performance_summaries, feed_dict={self.tf_loss: eval_dict["loss"],
                                                                      self.tf_train_accuracy: eval_dict["train_accuracy"],
                                                                      self.tf_train_error: eval_dict["train_error"],
                                                                      self.tf_valid_accuracy: eval_dict["valid_accuracy"],
                                                                      self.tf_valid_error: eval_dict["valid_error"]})

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()
