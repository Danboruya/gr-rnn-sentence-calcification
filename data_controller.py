import re
import numpy as np
import os
from tensorflow.contrib import learn


class DataSet:
    def __init__(self):
        self.positive_data = []
        self.negative_data = []
        self.all_data_set = []
        self.data_set_label = []


def _string_cleaner(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def _load_data(positive_data_path, negative_data_path):
    """
    Loading a data from data files.
    :param positive_data_path: Positive data file path.
    :param negative_data_path: Negative data file path.
    :return: Dataset class object
    """
    data_set = DataSet()

    # Load files
    with open(positive_data_path, 'r') as raw_positive_data, \
            open(negative_data_path, 'r') as raw_negative_data:
        raw_positive_data.readline()
        raw_negative_data.readline()
        raw_positive_sentences = [s.strip() for s in raw_positive_data]
        raw_negative_sentences = [s.strip() for s in raw_negative_data]

    # Formatting data
    data_set.positive_data = [_string_cleaner(sentence) for sentence in raw_positive_sentences]
    data_set.negative_data = [_string_cleaner(sentence) for sentence in raw_negative_sentences]
    data_set.all_data_set = data_set.positive_data + data_set.negative_data

    positive_labels = [[0, 1] for _ in data_set.positive_data]
    negative_labels = [[1, 0] for _ in data_set.negative_data]
    data_set.data_set_label = np.concatenate([positive_labels, negative_labels], 0)

    return data_set


def load_data_file(pos_data_file_path, neg_data_file_path):
    """
    Loading datafiles with file path
    :param pos_data_file_path: Positive data files path
    :param neg_data_file_path: Negative data files path
    :return: Data set class object
    """
    return _load_data(pos_data_file_path, neg_data_file_path)


def build_vocabulary(positive_data, negative_data, all_data_set):
    """
    Build vocabulary data
    :param positive_data: Positive sentence data
    :param negative_data: Negative sentence data
    :param all_data_set: All sentence data
    :return: Vocabulary data and input data
    """
    positive_max_sentence_length = max([len(sentence.split(" ")) for sentence in positive_data])
    negative_max_sentence_length = max([len(sentence.split(" ")) for sentence in negative_data])
    data_set_max_sentence_length = max([len(sentence.split(" ")) for sentence in all_data_set])

    # Build vocabulary from all data
    vocab_processor = learn.preprocessing.VocabularyProcessor(data_set_max_sentence_length)
    x = np.array(list(vocab_processor.fit_transform(all_data_set)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])

    # Build vocabulary from positive data
    pos_vocab_processor = learn.preprocessing.VocabularyProcessor(positive_max_sentence_length)
    x_pos = np.array(list(pos_vocab_processor.fit_transform(positive_data)))
    x_pos_as_x = np.array(list(vocab_processor.fit_transform(positive_data)))
    pos_vocab_dict = pos_vocab_processor.vocabulary_._mapping
    sorted_pos_vocab = sorted(pos_vocab_dict.items(), key=lambda x_pos: x_pos[1])
    pos_vocabulary = list(list(zip(*sorted_pos_vocab))[0])

    # Build vocabulary from negative data
    neg_vocab_processor = learn.preprocessing.VocabularyProcessor(negative_max_sentence_length)
    x_neg = np.array(list(neg_vocab_processor.fit_transform(negative_data)))
    x_neg_as_x = np.array(list(vocab_processor.fit_transform(negative_data)))
    neg_vocab_dict = neg_vocab_processor.vocabulary_._mapping
    sorted_neg_vocab = sorted(neg_vocab_dict.items(), key=lambda x_neg: x_neg[1])
    neg_vocabulary = list(list(zip(*sorted_neg_vocab))[0])

    # Formatting data
    vocab_data = [vocabulary, pos_vocabulary, neg_vocabulary, vocab_processor]
    input_data = [x, x_pos, x_neg, x_pos_as_x, x_neg_as_x, data_set_max_sentence_length]

    return [vocab_data, input_data]


def data_divider(data_set, label_set):
    """
    Divide the data for cross validation.
    This method provide 10% train data form argument data set.
    Argument data sets are shuffled by random seed 10.
    :param data_set: Target data set
    :param label_set: Target data Label list
    :return: Array of train/test data set and train/test label
    """
    counter = 0
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    # Shuffle operation
    np.random.seed(10)
    shuffle_index = np.random.permutation(len(data_set))
    shuffled_data_set = data_set[shuffle_index]
    shuffled_label_set = label_set[shuffle_index]

    for sentence in shuffled_data_set:
        if counter < (len(shuffled_data_set) - (len(shuffled_data_set) * 0.01)):
            train_data.append(sentence)
            counter += 1
        else:
            test_data.append(sentence)
            counter += 1
    counter = 0
    for label in shuffled_label_set:
        if counter < (len(shuffled_label_set) - (len(shuffled_label_set) * 0.01)):
            train_label.append(label)
            counter += 1
        else:
            test_label.append(label)
            counter += 1
    sample_data = [train_data, test_data]
    sample_label = [train_label, test_label]
    return [sample_data, sample_label]
