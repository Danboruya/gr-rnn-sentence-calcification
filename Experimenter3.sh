#! /bin/bash

# Train model and Test model

# Experiment 3 : Change n_layer : Default=2
python sentence_classification.py --cell_type="RNN" --n_layer=1;
python sentence_classification.py --cell_type="RNN" --n_layer=2;
python sentence_classification.py --cell_type="RNN" --n_layer=3;
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=1;
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=2;
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=3;
python sentence_classification.py --cell_type="LSTM" --n_layer=1;
python sentence_classification.py --cell_type="LSTM" --n_layer=2;
python sentence_classification.py --cell_type="LSTM" --n_layer=3;
python sentence_classification.py --cell_type="GRU" --n_layer=1;
python sentence_classification.py --cell_type="GRU" --n_layer=2;
python sentence_classification.py --cell_type="GRU" --n_layer=3;