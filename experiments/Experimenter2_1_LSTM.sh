#! /bin/bash

# Experiment 2 : Change Hyper-parameters
# Experiment 2-1 : Change learning_rate : Default 1e-6
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=16 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python ../sentence_classification.py --cell_type="LSTM" --learning_rate=1e-6 --n_cell=64 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";
python ../sentence_classification.py --cell_type="PLSTM" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp1";

echo "Experiment 2-1 on LSTM cell has been complected."
