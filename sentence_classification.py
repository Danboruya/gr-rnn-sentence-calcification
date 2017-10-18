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
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

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

# ========
# Training
# ========
