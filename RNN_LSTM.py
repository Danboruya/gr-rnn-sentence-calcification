import tensorflow as tf


class RnnWithLstm(object):
    """
    Sentence classification on RNN with LSTM.
    """
    def __init__(self, sentence_length, n_class, vocab_size,
                 embedding_size, n_unit, batch_size):
        """
        Initialize the instance of this class.
        :param sentence_length: Max length of sentence
        :param n_class: The number of class for classier
        :param vocab_size: The number of vocabulary
        :param embedding_size: Word embedding size
        :param n_unit: The number of unit on LSTM cell
        :param batch_size: Size of batch
        """

        # Set Placeholders for input layer, output layer and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sentence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, n_class], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_prob")

        # Word embedding layer as input layer
        with tf.device('/cpu:0'), tf.name_scope("word_embedding"):
            self.w = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                                   -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.w, self.input_x)
            self.embedded_chars_exp = tf.expand_dims(self.embedded_chars, -1)
        print("Embedding: Done")

        # Create RNN as hidden layer
        with tf.name_scope("RNN_Cell"):
            # For test creation, now we provide only basic lstm cell
            cell = tf.contrib.rnn.BasicLSTMCell(n_unit, forget_bias=1.0)
            # self.initial_state = cell.zero_state(batch_size, tf.float32)
            self.initial_state = cell.zero_state(tf.shape(self.input_x)[0], tf.float32)
            print(self.initial_state)
            self.state = self.initial_state
            outputs = []
            with tf.variable_scope('LSTM'):
                for time_step in range(sentence_length):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    cell_output, self.state = cell(self.embedded_chars[:, time_step, :], self.state)
                    outputs.append(cell_output)
            self.output = outputs[-1]
            # self.output = tf.concat(output, 1)
            # self.h_out = tf.reshape(self.output, [-1, n_unit])
        print("RNN: Done")

        # Dropout in hidden layer
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.output, keep_prob=self.dropout_keep_prob)
        print("Dropout: Done")

        # Output layer
        with tf.name_scope("output"):
            w = tf.get_variable("w", [n_unit, n_class])
            b = tf.get_variable("b", [n_class])
            self.scores = tf.nn.xw_plus_b(self.h_drop, w, b, name="scores")
            self.predictions = tf.argmax(self.scores, axis=1, name="predictions")
        print("Output: Done")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses)
        print("Loss: Done")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        print("Accuracy: Done")
        print("==Network construction has been finished==")
