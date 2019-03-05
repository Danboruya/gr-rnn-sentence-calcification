#!/usr/bin/env bash

# Experiment 3 [Change the number of hidden layer] [LSTM, PLSTM]
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_layer=1 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_1layer";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_layer=2 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_2layer";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_3layer";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_layer=1 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_1layer";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_layer=2 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_2layer";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp03_3layer";

echo "Experiment 3: \"Change the number of hidden layer [LSTM, PLSTM]\" has been completed."
