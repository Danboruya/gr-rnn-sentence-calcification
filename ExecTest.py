import data_loader


def main():
    POS_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.pos"
    NEG_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.neg"
    data_set = data_loader.load_data_file(POS_FILE_PATH, NEG_FILE_PATH)

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


if __name__ == "__main__":
    main()
