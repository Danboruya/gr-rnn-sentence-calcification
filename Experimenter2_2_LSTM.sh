#! /bin/bash

# Experiment 2 : Change Hyper-parameters
# Experiment 2-2 : Change n_layer : Default=2
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=1 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=2 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="BasicLSTM" --n_layer=3 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="LSTM" --n_layer=1 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="LSTM" --n_layer=2 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="LSTM" --n_layer=3 --learning_rate=1e-3 --n_cell=32 --n_epoch=500 --dropout_keep_prob=0.7 --exp_name="exp2_2";