import data_controller


def main():
    POS_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.pos"
    NEG_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.neg"
    data_set = data_controller.load_data_file(POS_FILE_PATH, NEG_FILE_PATH)

    # print("Positive data")
    print(data_set.positive_data)
    # print("=============")
    # print("Negative data")
    print(data_set.negative_data)
    print("=============")
    print("Data set information")
    print("The number of data set : " + str(len(data_set.positive_data) + len(data_set.negative_data)))
    print("Positive data : " + str(len(data_set.positive_data)))
    print("Negative data : " + str(len(data_set.negative_data)))
    print("=============")

    vocab_data, input_data = data_controller.build_vocabulary(data_set.positive_data,
                                                              data_set.negative_data, data_set.all_data_set)

    print("=============")
    print("vocabulary information")
    print("Dataset vocabulary : " + str(len(vocab_data[0])))
    print("Positive vocabulary : " + str(len(vocab_data[1])))
    print("Negative vocabulary : " + str(len(vocab_data[2])))
    print("=============")
    print(vocab_data[0])
    print(vocab_data[1])
    print(vocab_data[2])
    print("=============")
    print("Formatted data information")
    print(input_data[0])
    print(input_data[3])
    print(input_data[4])
    print("=============")


if __name__ == "__main__":
    main()
