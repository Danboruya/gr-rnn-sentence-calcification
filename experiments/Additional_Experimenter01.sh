#! /bin/bash

# Train model and Test model

# Additional experiments [Increasing the hidden layer]
python ../sentence_classification.py --cell_type="RNN" --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer"
python ../sentence_classification.py --cell_type="LSTM" --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer";
python ../sentence_classification.py --cell_type="PLSTM" --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer";
python ../sentence_classification.py --cell_type="GRU" --n_layer=4 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --exp_name="add_exp01_4layer";

echo "Additional experiment \"Increasing hidden layer\" has been complected."
