import data_loader

def main():
    POS_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.pos"
    NEG_FILE_PATH = "./data/rt-polaritydata/rt-polarity-utf8.neg"
    in_text , y = data_loader.load_data_file(POS_FILE_PATH, NEG_FILE_PATH)

    # print(in_text)
    for label in y:
        print(label)

if __name__ == "__main__":
    main()