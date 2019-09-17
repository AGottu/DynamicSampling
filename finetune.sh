#!/bin/sh
pip install gdown
python $PWD/download_files.py
unzip -o $PWD/data/all_datasets.zip -d $PWD/data
python $PWD/download_drop.py
python -u $PWD/finetune_multi_dataset.py $1 $2 $3