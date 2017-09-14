import numpy as np
import re


def string_cleaner(string):
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


def load_data_and_category(positive_data, negative_data):
    """
    Loading a data of categorized sentences from data files.
    :param positive_data: Positive category data file
    :param negative_data: Negative category data file
    :return: Spited sentence and category
    """

    # Load the sentences from data files.
    # Step 1. Load file, Step 2. remove blank word
    # parameter pos_sentences : List of positive sentence
    # parameter neg_sentences : List of negative sentence
    pos_sentences = list(open(positive_data, "r").readlines())
    pos_sentences = [s.strip() for s in pos_sentences]
    neg_sentences = list(open(negative_data, "r").readlines())
    neg_sentences = [s.strip() for s in neg_sentences]

    # print(pos_sentences)
    # print("===========")
    # print(neg_sentences)

    # Create input sentence
    # Step 1. Combining 2 list, Step 2. Cleaning sentence format
    in_sentence = pos_sentences + neg_sentences
    in_sentence = [string_cleaner(sentence) for sentence in in_sentence]

    # print(in_sentence)

    # Create category labels
    # If positive sentence, labelled to [0, 1] as positive category.
    # If negative sentence, labelled to [1, 0] as negative category.
    pos_category = [[0, 1] for _ in pos_sentences]
    neg_category = [[1, 0] for _ in neg_sentences]

    # print(pos_category)
    # print(len(pos_category))
    # print("===========")
    # print(neg_category)
    # print(len(pos_category))

    # Set output format
    y = np.concatenate([pos_category, neg_category], axis=0)
    return [in_sentence, y]


def load_data_file(pos_data_file_path, neg_data_file_path):
    """
    Loading datafiles with file path
    :param pos_data_file_path: Positive data files path
    :param neg_data_file_path: Negative data files path
    :return: Categorized sentence
    """
    return load_data_and_category(pos_data_file_path, neg_data_file_path)
