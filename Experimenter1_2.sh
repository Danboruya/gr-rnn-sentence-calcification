#! /bin/bash

# Train model and Test model

# Experiment 1 : Change the number of cell
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-6 --n_cell=16 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python sentence_classification.py --cell_type="BasicLSTM" --learning_rate=1e-6 --n_cell=64 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=16 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=64 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";

echo "Experiment 1 on BasicLSTM and LSTM cell has been complected."
