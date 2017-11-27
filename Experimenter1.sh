#! /bin/bash

# Train model and Test model

# Experiment 1 : Change cell type
python sentence_classification.py --cell_type="RNN" --n_cell=32 --n_epoch=500 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1"
python sentence_classification.py --cell_type="BasicLSTM" --n_cell=32 --n_epoch=500 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1"
python sentence_classification.py --cell_type="LSTM" --n_cell=32 --n_epoch=500 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1"
python sentence_classification.py --cell_type="GRU" --n_cell=32 --n_epoch=500 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1"