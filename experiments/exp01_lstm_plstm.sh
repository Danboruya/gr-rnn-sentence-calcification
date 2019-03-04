#!/usr/bin/env bash

# Experiment 1 [Change the learning rate] [LSTM, PLSTM]
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-3 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-3";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-4 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp_1e-4";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-5 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp_1e-5";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp_1e-6";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-7 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp_1e-7";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-3 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-3";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-4 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-4";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-5 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-5";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-6";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-7 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1_1e-7";

echo "Experiment 1: \"Change the learning rate [LSTM, PLSTM]\" has been completed."
