import tflearn as tfl

# The test model for LSTM with tflearn library


class TextRnnWithLstm:
    def __init__(self):
        self.net = None
        self.train_data = []
        self.train_data_label = []
        self.test_data = []
        self.test_data_label = []


def build_neural_network(data_set, data_set_label):
    # Set variables
    t_rnn = TextRnnWithLstm()
    t_rnn.train_data = data_set[0]
    t_rnn.test_data = data_set[1]
    t_rnn.train_data_label = data_set_label[0]
    t_rnn.test_data_label = data_set_label[1]

    # Build network
    # Input layer
    net = tfl.input_data([None, len(t_rnn.train_data)])

    # Hidden layer
    # Word embedding layer
    net = tfl.embedding(net, input_dim=100000, output_dim=128)
    net = tfl.lstm(net, 64, activation='sigmoid')

    # Output layer
    net = tfl.fully_connected(net, 2, activation='softmax')

    # Learning condition
    net = tfl.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    t_rnn.net = net

    return t_rnn


def train(t_rnn):
    net = t_rnn.net
    model = tfl.DNN(net, tensorboard_verbose=3)
    model.fit(t_rnn.train_data, t_rnn.train_data_label, validation_set=(t_rnn.train_data, t_rnn.train_data_label),
              show_metric=True, batch_size=32, run_id="lstm_model")


def test(t_rnn):
    net = t_rnn.net
    model = tfl.DNN(net, tensorboard_verbose=3)
    model.fit(t_rnn.test_data, t_rnn.test_data_label, validation_set=(t_rnn.test_data, t_rnn.test_data_label),
              show_metric=True, batch_size=32, run_id="lstm_model")
