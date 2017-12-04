#! /bin/bash

# Experiment 2 : Change Hyper-parameters
# Experiment 2-2 : Change n_layer : Default=2
python sentence_classification.py --cell_type="RNN" --n_layer=1 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="RNN" --n_layer=2 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="RNN" --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="GRU" --n_layer=1 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="GRU" --n_layer=2 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";
python sentence_classification.py --cell_type="GRU" --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="exp2_2";

echo "Experiment 2-2 on RNN and GRU cell has been complected."
