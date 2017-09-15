import re


class DataSet:
    def __init__(self):
        self.positive_data = []
        self.negative_data = []


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

    return data_set


def load_data_file(pos_data_file_path, neg_data_file_path):
    """
    Loading datafiles with file path
    :param pos_data_file_path: Positive data files path
    :param neg_data_file_path: Negative data files path
    :return: Data set class object
    """
    return _load_data(pos_data_file_path, neg_data_file_path)
