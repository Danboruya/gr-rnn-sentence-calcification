#! /bin/bash

# Train model and Test model

# Additional experiments [Increasing the number of epoch for GRU and CNN cell]
python sentence_classification.py --cell_type="GRU" --n_layer=3 --n_cell=32 --n_epoch=600 --dropout_keep_prob=0.7 --exp_name="add_exp01_3layer_600epoch"
python sentence_classification.py --cell_type="CNN" --n_epoch=600 --n_layer=1 --dropout_keep_prob=0.7 --exp_name="add_exp01_3layer_600epoch";


echo "Additional experiment \"Increasing the number of epoch for GRU and CNN cell\" has been complected."
