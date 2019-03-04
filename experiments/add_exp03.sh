#!/usr/bin/env bash

# Additional experiments 3: [Decreasing the word embeddings dimension to 128]
python ./sentence_classification.py --cell_type="RNN" --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="add_exp03_128dim";
python ./sentence_classification.py --cell_type="LSTM" --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="add_exp03_128dim";
python ./sentence_classification.py --cell_type="PLSTM" --n_layer=3 --n_cell=32 --n_epoch=400 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="add_exp03_128dim";
python ./sentence_classification.py --cell_type="GRU" --n_epoch=400 --n_layer=3 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="add_exp03_128dim";
python ./sentence_classification.py --cell_type="CNN" --n_layer=1 --dropout_keep_prob=0.7 --embedding_dim=128 --exp_name="add_exp03_128dim";

echo "Additional experiment 3: \"Decreasing the word embeddings dimension to 128\" has been complected."
