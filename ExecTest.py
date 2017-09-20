import os
import numpy as np
from tensorflow.contrib import learn
import data_controller


def main():
    POS_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.pos"
    NEG_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.neg"
    data_set = data_controller.load_data_file(POS_FILE_PATH, NEG_FILE_PATH)

    print("Positive data")
    print(data_set.positive_data)
    print("=============")
    print("Negative data")
    print(data_set.negative_data)
    print("=============")
    print("Data set information")
    print("The number of data set : " + str(len(data_set.positive_data) + len(data_set.negative_data)))
    print("Positive data : " + str(len(data_set.positive_data)))
    print("Negative data : " + str(len(data_set.negative_data)))
    print("=============")

    build_vocabulary(data_set)


def build_vocabulary(data_set):
    """
    Building vocabulary from dataset
    :param data_set: Dataset object
    """
    var_word = ""
    pos_max_docment_length = max([len(sentence.split(" ")) for sentence in data_set.positive_data])
    neg_max_docment_length = max([len(sentence.split(" ")) for sentence in data_set.negative_data])
    neg_vocab = []

    print(pos_max_docment_length)
    print(neg_max_docment_length)

    # Build positive data vocabulary
    max_document_length = max([len(x.split(" ")) for x in data_set.all_data_set])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(data_set.all_data_set)))
    # for statement in x:
    #    print(statement)
    print("Vocab_size: " + str(len(vocab_processor.vocabulary_)))
    vocab_dict = vocab_processor.vocabulary_._mapping
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    vocabulary = list(list(zip(*sorted_vocab))[0])
    print(vocabulary)


if __name__ == "__main__":
    main()
