import tensorflow as tf
import os
import time
import datetime
import data_controller
from RNN_LSTM import RnnWithLstm

# ==================
# Parameter settings
# ==================

flags = tf.flags

# ==Data==
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity-utf8.pos",
                    "Data source for the positive data.")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity-utf8.neg",
                    "Data source for the negative data.")

# ==Hyper parameters==
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of word embedding")
flags.DEFINE_float("n_unit", 2, "The number of unit of lstm")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
flags.DEFINE_integer("n_class", 2, "The number of classifier")

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
sentence_length = raw_input_data[5]
n_class = FLAGS.n_class
vocab_processor = vocab_data[3]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_test)))

# ========
# Training
# ========
with tf.Graph().as_default():
    session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                    log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_config)
    with sess.as_default():
        rnn_lstm = RnnWithLstm(
            sentence_length=sentence_length,
            n_class=n_class,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            n_unit=FLAGS.n_unit,
            batch_size=FLAGS.batch_size
        )

        print("Network instance has been created")

        # Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("Optimizer has been set")
        grads_and_vars = optimizer.compute_gradients(rnn_lstm.loss)
        train_optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        print("Complected training procedure set")

        # Define summaries settings for TensorBoard
        gradient_summaries = []
        for grad, var in grads_and_vars:
            if grad is not None:
                gradient_histogram_summary = tf.summary.histogram("{}/gradient/histogram".format(var.name), grad)
                sparsity_summary = tf.summary.scalar("{}/gradient/sparsity".format(var.name), tf.nn.zero_fraction(grad))
                gradient_summaries.append(gradient_histogram_summary)
                gradient_summaries.append(sparsity_summary)
        gradient_summaries_merged = tf.summary.merge(gradient_summaries)

        # Output directory for model and summaries
        timestamp = str(int(time.time()))
        output_directory = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}".format(output_directory))

        # Loss and accuracy summary
        loss_summary = tf.summary.scalar("loss", rnn_lstm.loss)
        accuracy_summary = tf.summary.scalar("accuracy", rnn_lstm.accuracy)

        # Train summaries
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary, gradient_summaries_merged])
        train_summary_dir = os.path.join(output_directory, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        print("Train summary has been set")

        # Test summaries
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_dir = os.path.join(output_directory, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)
        print("Test summary has been set")

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(output_directory, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        print("Checkpoint directory has been set")

        # Save vocabulary
        vocab_processor.save(os.path.join(output_directory, "vocab"))
        print("Vocabulary has been saved")

        # Initialize all variables for tensorflow
        sess.run(tf.global_variables_initializer())
        print("Boot Session")


        def train_step(x_batch, y_batch):
            """
            A single training step
            :param x_batch: Batch for input data
            :param y_batch: Batch for output data
            """
            feed_dict = {
                rnn_lstm.input_x: x_batch,
                rnn_lstm.input_y: y_batch,
                rnn_lstm.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_optimizer, global_step, train_summary_op, rnn_lstm.loss, rnn_lstm.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            train_summary_writer.add_summary(summaries, step)
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a test set
            :param x_batch: Batch for input data
            :param y_batch: Batch for label data
            :param writer: Summary writer
            """

            feed_dict = {
                rnn_lstm.input_x: x_batch,
                rnn_lstm.input_y: y_batch,
                rnn_lstm.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, test_summary_op, rnn_lstm.loss, rnn_lstm.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_controller.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.n_epoch)
        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                test_step(x_test, y_test, writer=test_summary_writer)
                print("===============")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))


