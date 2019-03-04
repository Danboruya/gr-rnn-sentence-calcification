#!/usr/bin/env bash

# Experiment 2 [Change the number of cell] [LSTM, PLSTM]
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=16 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_16cells";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_32cells";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=64 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_64cells";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=16 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_16cells";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_32cells";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=64 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_64cells";

echo "Experiment 2: \"Change the number of cell [LSTM, PLSTM]\" has been completed."
