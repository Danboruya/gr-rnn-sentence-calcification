import data_controller


def main():
    POS_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.pos"
    NEG_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.neg"
    raw_data_set = data_controller.load_data_file(POS_FILE_PATH, NEG_FILE_PATH)
    vocab_data, raw_input_data = data_controller.build_vocabulary(raw_data_set.positive_data,
                                                                  raw_data_set.negative_data, raw_data_set.all_data_set)
    # data_set/label[0]:Train, data_set/label[1]:Test
    data_set, data_set_label = data_controller.data_divider(raw_input_data[0], raw_data_set.data_set_label)

    print_log(raw_data_set, vocab_data, raw_input_data, data_set, data_set_label)


def print_log(raw_data_set, vocab_data, raw_input_data, data_set, data_set_label):
    # print("Positive data")
    # print(raw_data_set.positive_data)
    # print("=============")
    # print("Negative data")
    # print(raw_data_set.negative_data)
    # print("=============")
    # print(vocab_data[0])
    # print(vocab_data[1])
    # print(vocab_data[2])
    # print("=============")
    # print("Formatted data information")
    # print("Input X:")
    # print(raw_input_data[0])
    # print("Input X only pos:")
    # print(raw_input_data[3])
    # print("Input X only neg:")
    # print(raw_input_data[4])
    # print("Test data")
    # for test_data in data_set[0]:
    #     print(test_data)
    # print(len(data_set[0]))
    # print(len(data_set[1]))
    # print(len(data_set[0])+len(data_set[1]))
    print("vocabulary information")
    print("Dataset vocabulary : " + str(len(vocab_data[0])))
    print("Positive vocabulary : " + str(len(vocab_data[1])))
    print("Negative vocabulary : " + str(len(vocab_data[2])))
    print("=============")
    print("Data set information")
    print("The number of data set : " + str(len(raw_data_set.positive_data) + len(raw_data_set.negative_data)))
    print("Positive data : " + str(len(raw_data_set.positive_data)))
    print("Negative data : " + str(len(raw_data_set.negative_data)))
    print("=============")


if __name__ == "__main__":
    main()
