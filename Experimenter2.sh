#! /bin/bash

# Experiment 2 : Change Hyper-parameters
# Experiment 2-1 : Change learning_rate : Default 1e-6
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-3;
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-4;
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-5;
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-6;
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-7;
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-3;
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-4;
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-5;
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-6;
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-7;
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-3;
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-4;
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-5;
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6;
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-7;
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-3;
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-4;
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-5;
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-6;
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-7;

# Experiment 2-2 : Change dropout_keep_prob : Default=0.5
python sentence_classification.py --cell_type="RNN" --dropout_keep_prob=0.8;
python sentence_classification.py --cell_type="RNN" --dropout_keep_prob=0.6;
python sentence_classification.py --cell_type="RNN" --dropout_keep_prob=0.5;
python sentence_classification.py --cell_type="BasicLSTM" --dropout_keep_prob=0.8;
python sentence_classification.py --cell_type="BasicLSTM" --dropout_keep_prob=0.6;
python sentence_classification.py --cell_type="BasicLSTM" --dropout_keep_prob=0.5;
python sentence_classification.py --cell_type="LSTM" --dropout_keep_prob=0.8;
python sentence_classification.py --cell_type="LSTM" --dropout_keep_prob=0.6;
python sentence_classification.py --cell_type="LSTM" --dropout_keep_prob=0.5;
python sentence_classification.py --cell_type="GRU" --dropout_keep_prob=0.8;
python sentence_classification.py --cell_type="GRU" --dropout_keep_prob=0.6;
python sentence_classification.py --cell_type="GRU" --dropout_keep_prob=0.5;

# Experiment 2-3 : Change n_cell : Default=32
python sentence_classification.py --cell_type="RNN" --n_cell=16;
python sentence_classification.py --cell_type="RNN" --n_cell=32;
python sentence_classification.py --cell_type="RNN" --n_cell=64;
python sentence_classification.py --cell_type="BasicLSTM" --n_cell=16;
python sentence_classification.py --cell_type="BasicLSTM" --n_cell=32;
python sentence_classification.py --cell_type="BasicLSTM" --n_cell=64;
python sentence_classification.py --cell_type="LSTM" --n_cell=16;
python sentence_classification.py --cell_type="LSTM" --n_cell=32;
python sentence_classification.py --cell_type="LSTM" --n_cell=64;
python sentence_classification.py --cell_type="GRU" --n_cell=16;
python sentence_classification.py --cell_type="GRU" --n_cell=32;
python sentence_classification.py --cell_type="GRU" --n_cell=64;