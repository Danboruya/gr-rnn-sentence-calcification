#! /bin/bash

# Train model and Test model

# Experiment 0 : CNN model
python ../sentence_classification.py --cell_type="CNN" --learning_rate=1e-3 --n_epoch=200 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="exp0_1e-3";
python ../sentence_classification.py --cell_type="CNN" --learning_rate=1e-4 --n_epoch=200 --n_layer=1 --dropout_keep_prob=0.8 --exp_name="exp0_1e-4";
python ../sentence_classification.py --cell_type="CNN" --learning_rate=1e-6 --n_epoch=200 --n_layer=1 --dropout_keep_prob=0.8 --exp_name="exp0_1e-6";


echo "Experiment 0: \"CNN model\" has been completed."
