#! /bin/bash

# Experiment 2 : Change Hyper-parameters
# Experiment 2-1 : Change learning_rate : Default 1e-6
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-3 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-4 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-5 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="RNN" --learning_rate=1e-7 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-3 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-4 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-5 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-6 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";
python sentence_classification.py --cell_type="GRU" --learning_rate=1e-7 --n_cell=32 --n_epoch=400 --n_layer=2 --dropout_keep_prob=0.7 --exp_name="exp2_1";

echo "Experiment 2-1 on RNN and GRU cell has been complected."
