import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import os
import time
import csv
import datetime
import data_controller
from Model import Model

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
flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of word embedding")
flags.DEFINE_integer("n_cell", 32, "The number of unit of cell")
flags.DEFINE_integer("n_layer", 2, "The number of hidden layer")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
flags.DEFINE_float("learning_rate", 1e-6, "Learning rate")
flags.DEFINE_integer("n_class", 2, "The number of classifier")
flags.DEFINE_float("f_bias", 1.0, "Forget bias")
flags.DEFINE_string("cell_type", "GRU", "The type of cell on hidden layer")
flags.DEFINE_string("filter_sizes", "3,4,4", "The size of cnn filter")
flags.DEFINE_integer("n_filter", 4, "The number of filter par filter size")

# ==Training parameters==
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_integer("n_epoch", 300, "The number of training epochs")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps")
flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")

# ==Other parameters==
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
flags.DEFINE_boolean("enable_data_check", False, "Only run checking dataset function")
flags.DEFINE_string("exp_name", "exp1", "Experiment name")

FLAGS = flags.FLAGS


def load_data():
    """
    Loading the data from data set
    :return: Training components
    """
    # Data preparation
    raw_data_set = data_controller.load_data_file(FLAGS.positive_data_file, FLAGS.negative_data_file)
    vocab_data, raw_input_data = data_controller.build_vocabulary(raw_data_set.positive_data,
                                                                  raw_data_set.negative_data, raw_data_set.all_data_set)
    # data_set/label[0]:Train, data_set/label[1]:Test
    data_set, data_set_label = data_controller.data_divider(raw_input_data[0], raw_data_set.data_set_label)

    x_train = data_set[0]
    y_train = data_set_label[0]
    x_valid = data_set[1]
    y_valid = data_set_label[1]
    sentence_length = raw_input_data[5]
    n_class = FLAGS.n_class
    vocab_processor = vocab_data[3]

    print_info(vocab_processor, y_train, y_valid)

    components = [x_train, y_train, x_valid, y_valid, sentence_length, n_class, vocab_processor]
    return components


def load_test_data(out_dir):
    """
    Load data for test
    :param out_dir: Output directory
    :return: Test components
    """
    print("===================")
    print("Evaluating process")
    # Test data. Test label is; 0:Negative, 1:Positive
    x_raw = ["a masterpiece four years in the making",
             "everything is off.",
             "I agree with you.",
             "That's to bad",
             "I succeeded in uninstalling"]
    y_test = [1, 0, 1, 0, 1]
    vocab_path = os.path.join(out_dir, "vocab")
    print("Vocab_path" + vocab_path)
    print("Output dir" + out_dir)
    test_vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(test_vocab_processor.transform(x_raw)))
    test_components = [x_test, y_test, x_raw]
    return test_components


def print_info(vocab_processor, y_train, y_valid):
    print("===================")
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)))
    print("Total dataset entities: {:d}".format(len(y_train) + len(y_valid)))
    print("Parameters:")
    for key in FLAGS.__flags.keys():
        print('{}={}'.format(key, getattr(FLAGS, key)))
    print("===================")


def train(x_train, y_train, x_valid, y_valid, sentence_length, n_class, vocab_processor):
    """
    Train process.
    :param x_train: Input data for training
    :param y_train: Label data for training
    :param x_valid: Input data for validating
    :param y_valid: Label data for validating
    :param sentence_length: Sentence length of input data
    :param n_class: The number of classifier
    :param vocab_processor: Vocabulary processor
    :return:
    """
    with tf.Graph().as_default():
        session_config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                        log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_config)
        with sess.as_default():
            net = Model(
                sentence_length=sentence_length,
                n_class=n_class,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                n_unit=FLAGS.n_cell,
                n_layer=FLAGS.n_layer,
                cell_type=FLAGS.cell_type,
                f_bias=FLAGS.f_bias,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                n_filter=FLAGS.n_filter
            )

            print("Network instance has been created")

            # Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            print("Optimizer has been set")
            grads_and_vars = optimizer.compute_gradients(net.loss)
            train_optimizer = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            print("Complected training procedure set")

            # Define summaries settings for TensorBoard
            gradient_summaries = []
            for grad, var in grads_and_vars:
                if grad is not None:
                    gradient_histogram_summary = tf.summary.histogram("{}/gradient/histogram".format(var.name), grad)
                    sparsity_summary = tf.summary.scalar("{}/gradient/sparsity".format(var.name),
                                                         tf.nn.zero_fraction(grad))
                    gradient_summaries.append(gradient_histogram_summary)
                    gradient_summaries.append(sparsity_summary)
            gradient_summaries_merged = tf.summary.merge(gradient_summaries)

            # Output directory for model and summaries
            timestamp = str(int(time.time()))
            output_directory = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.cell_type +
                                                            "_" + FLAGS.exp_name + "_" + timestamp))
            print("Writing to {}".format(output_directory))

            # Loss and accuracy summary
            loss_summary = tf.summary.scalar("loss", net.loss)
            accuracy_summary = tf.summary.scalar("accuracy", net.accuracy)

            # Train mini batch summaries
            train_mini_batch_summary_op = tf.summary.merge([loss_summary, accuracy_summary, gradient_summaries_merged])
            train_mini_batch_summary_dir = os.path.join(output_directory, "summaries", "train_mini_bach")
            train_mini_batch_summary_writer = tf.summary.FileWriter(train_mini_batch_summary_dir, sess.graph)
            print("Train mini batch summary has been set")

            # Validate summaries
            validate_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            validate_summary_dir = os.path.join(output_directory, "summaries", "validate")
            validate_summary_writer = tf.summary.FileWriter(validate_summary_dir, sess.graph)
            print("Test summary has been set")

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(output_directory, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            print("Checkpoint file write to {}".format(checkpoint_dir))
            print("Checkpoint directory has been set")

            # Save vocabulary
            vocab_processor.save(os.path.join(output_directory, "vocab"))
            print("Vocabulary has been saved")

            # Out path for result of validation
            valid_out_path = os.path.join(output_directory, "valid.log")

            # Initialize all variables for tensorflow
            sess.run(tf.global_variables_initializer())
            print("Boot Session")

            def train_step(x_batch, y_batch, current_epoch):
                """
                A single training step
                :param x_batch: Batch for input data
                :param y_batch: Batch for output data
                :param current_epoch: Current epoch
                """
                feed_dict = {
                    net.input_x: x_batch,
                    net.input_y: y_batch,
                    net.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, _train_step, _train_summaries, _train_loss, _train_accuracy, output = sess.run(
                    [train_optimizer, global_step, train_mini_batch_summary_op, net.loss,
                     net.accuracy, net.output],
                    feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                train_mini_batch_summary_writer.add_summary(_train_summaries, _train_step)
                print("step {}, epoch {}, loss {:g}, accuracy {:g}".format(_train_step, current_epoch,
                                                                           _train_loss, _train_accuracy))

            def valid_step(_x_valid, _y_valid, writer=None):
                """
                Validate model on a validation set
                :param _x_valid: input data
                :param _y_valid: label data
                :param writer: Summary writer
                """
                feed_dict = {
                    net.input_x: _x_valid,
                    net.input_y: _y_valid,
                    net.dropout_keep_prob: 1.0
                }
                _step, _summaries, _loss, _accuracy = sess.run(
                    [global_step, validate_summary_op, net.loss, net.accuracy],
                    feed_dict)
                _time_str = datetime.datetime.now().isoformat()
                if writer:
                    writer.add_summary(_summaries, _step)
                return [_time_str, _step, _loss, _accuracy]

            # Training loop
            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_valid = np.array(x_valid)
            y_valid = np.array(y_valid)
            n_data = len(x_train)
            current_step = 0
            for epoch in range(FLAGS.n_epoch):
                sff_idx = np.random.permutation(n_data)
                for idx in range(0, n_data, FLAGS.batch_size):
                    batch_x = x_train[
                        sff_idx[idx: idx + FLAGS.batch_size if idx + FLAGS.batch_size < n_data else n_data]]
                    batch_t = y_train[
                        sff_idx[idx: idx + FLAGS.batch_size if idx + FLAGS.batch_size < n_data else n_data]]
                    if len(batch_t) is FLAGS.batch_size:
                        train_step(batch_x, batch_t, epoch)
                        current_step = tf.train.global_step(sess, global_step)
                if epoch == 0 or epoch % 5 == 0 or epoch == (FLAGS.n_epoch - 1):
                    print("============")
                    print("Validation")
                    # train_step(x_train, y_train, epoch, is_batch=False)
                    time_str, step, loss, accuracy = valid_step(x_valid, y_valid, writer=validate_summary_writer)
                    print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
                    with open(valid_out_path, 'a+') as f:
                        f.write("============" + "\n")
                        f.write("Validation" + "\n")
                        f.write("{}: step {}, loss {:g}, accuracy {:g}, epoch: {:d}".format(time_str, step, loss,
                                                                                            accuracy, epoch) + "\n")
                        f.write("============" + "\n")
                        f.write("\n")
                    print("============")
                # if current_step % FLAGS.checkpoint_every == 0:
                if epoch % FLAGS.checkpoint_every == 0 or epoch == (FLAGS.n_epoch - 1):
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}".format(path))
                # saver.save(sess, os.path.join(output_directory, "model.ckpt"), global_step=current_step)
            output_directories = [output_directory, checkpoint_dir]
            return output_directories


def test(x_test, y_test, x_raw, out_dir, check_dir, vocab_processor, y_train, y_valid):
    """
    Test process
    :param x_test: Input data for testing
    :param y_test: Label data for testing
    :param x_raw: Original sentence of test data
    :param out_dir: Output directory path
    :param check_dir: Check point directory path
    :param vocab_processor: Vocabulary processor object
    :param y_train: Label data for training
    :param y_valid: Label data for validating
    """
    output_dir_path = out_dir
    checkpoint_file = tf.train.latest_checkpoint(check_dir)
    print("Checkpoint path: " + check_dir)
    graph = tf.Graph()
    print("Evaluate model")
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Restore the model
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("Output_layer/predictions").outputs[0]
            all_predictions = []

            # Classification process
            test_data = data_controller.test_data_provider(x_test, FLAGS.batch_size)
            for sentence in test_data:
                sentence_predictions = sess.run(predictions, {input_x: sentence, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, sentence_predictions])
    if y_test is not None:
        # correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test data: {}".format(len(y_test)))
        # print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
    csv_out_path = os.path.join(output_dir_path, "prediction.csv")
    for idx in range(len(predictions_human_readable)):
        print("Classification : " + str(predictions_human_readable[idx]) + ", Correct Label: " + str(y_test[idx]))
    print("Saving evaluation to {0}".format(csv_out_path))
    with open(csv_out_path, "w+") as f:
        csv.writer(f).writerows(predictions_human_readable)
    spec_out_path = os.path.join(output_dir_path, "spec.txt")
    with open(spec_out_path, "w+") as f:
        f.write("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)) + "\n")
        f.write("Train/Valid split: {:d}/{:d}".format(len(y_train), len(y_valid)) + "\n")
        f.write("Parameters:")
        for _attribute, _value in sorted(FLAGS.__flags.items()):
            f.write("{}={}".format(_attribute.upper(), _value) + "\n")


def run_test():
    """
    Test run for checking a dataset.
    """
    load_data()


def proc():
    """
    Main processes for this application.
    Load train data -> train -> Load test data -> test
    """
    components = load_data()
    out_dir = train(components[0], components[1], components[2],
                    components[3], components[4], components[5], components[6])
    test_components = load_test_data(out_dir[0])
    test(test_components[0], test_components[1], test_components[2], out_dir[0], out_dir[1],
         components[6], components[1], components[3])
    print_info(components[6], components[1], components[3])


def main():
    """
    Main function for this program.
    run_test(): A function for checking a dataset.
    proc(): A function for main process; Load train data -> Train -> Load test data -> test
    """
    if FLAGS.enable_data_check:
        run_test()
    else:
        proc()


if __name__ == "__main__":
    main()
