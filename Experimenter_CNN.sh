#! /bin/bash

# Train model and Test model

# Experiment 1 : Change the number of cell
python sentence_classification.py --cell_type="CNN" --learning_rate=1e-3 --n_epoch=200 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="exp_CNN_1";
python sentence_classification.py --cell_type="CNN" --learning_rate=1e-4 --n_epoch=200 --n_layer=1 --dropout_keep_prob=0.8 --exp_name="exp_CNN_2";
# python sentence_classification.py --cell_type="CNN" --learning_rate=1e-4 --n_epoch=400 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="exp_CNN_2";
# python sentence_classification.py --cell_type="CNN" --learning_rate=1e-6 --n_epoch=400 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="exp_CNN_3";


echo "Experiment CNN has been complected."
