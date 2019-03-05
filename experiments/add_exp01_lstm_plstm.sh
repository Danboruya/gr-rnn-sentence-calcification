#!/usr/bin/env bash

# Additional experiments 1: [Increasing the hidden layer] [LSTM, PLSTM]
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_layer=5 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_5layer";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_layer=5 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_5layer";

echo "Additional experiment 1: \"Increasing hidden layer\" has been complected. [LSTM, PLSTM]\" has been completed."
