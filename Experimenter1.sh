#! /bin/bash

# Train model and Test model

# Experiment 1 : Change cell type
python sentence_classification.py --cell_type="RNN"
python sentence_classification.py --cell_type="BasicLSTM"
python sentence_classification.py --cell_type="LSTM"
python sentence_classification.py --cell_type="GRU"