import tensorflow as tf
import os
import time
import datetime
import data_controller
from RNN_LSTM import RNNwithLSTM
from tensorflow.contrib import learn

# ==================
# Parameter settings
# ==================

flags = tf.flags

# ==Data==
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# ==Hyper parameters==
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding")
flags.DEFINE_float("n_unit", 2, "The number of unit of lstm")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")

# ==Training parameters==
flags.DEFINE_integer("batch_size", 64, "Batch size")
flags.DEFINE_integer("n_epoch", 100, "The number of training epochs")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")

# ==Other parameters==
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = flags.FLAGS
FLAGS._parse_flags()
print("Parameters:")
for attribute, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attribute.upper(), value))
print("===================")

# =============
# Prepossessing
# =============

# Data preparation
raw_data_set = data_controller.load_data_file(FLAGS.positive_data_file, FLAGS.negative_data_file)
vocab_data, raw_input_data = data_controller.build_vocabulary(raw_data_set.positive_data,
                                                              raw_data_set.negative_data, raw_data_set.all_data_set)
# data_set/label[0]:Train, data_set/label[1]:Test
data_set, data_set_label = data_controller.data_divider(raw_input_data[0], raw_data_set.data_set_label)

x_train = data_set[0]
y_train = data_set_label[0]
x_test = data_set[1]
y_test = data_set_label[1]
vocab_processor = vocab_data[3]

# ========
# Training
# ========

with tf.Graph().as_default():
    session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_config)
    with sess.as_default():
        rnn_lstm = RNNwithLSTM(
            sentence_length=x_train.shape[1],
            n_class=y_train.shape[1],
            vocab_size=vocab_processor,
            embedding_size=FLAGS.embedding_dim,
            n_unit=FLAGS.n_unit,
            batch_size=FLAGS.batch_size
        )

        # Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(rnn_lstm.loss)
        train_optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Define summaries settings for TensorBoard
        gradient_summaries = []
        for grad, var in grads_and_vars:
            if grad is not None:
                gradient_histogram_summary = tf.summary.histogram(("{}/gradient/histogram".format(var.name), grad))
                sparsity_summary = tf.summary.scalar("{}/gradient/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                gradient_summaries.append(gradient_histogram_summary)
                gradient_summaries.append(sparsity_summary)
        gradient_summaries_marged = tf.summary.merge(gradient_summaries)

        # Output directory for model and summaries
        timestamp = str(int(time.time()))
        output_directory = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(output_directory))

        # Loss and accuracy summary
        loss_summary = tf.summary.scalar("loss", rnn_lstm.loss)
        accuracy_summary = tf.summary.scalar("accuracy", rnn_lstm.accuracy)

        # Train summaries
