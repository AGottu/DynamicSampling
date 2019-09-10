#!/bin/sh
pip install gdown
python $PWD/download_files.py
unzip -o $PWD/data/all_datasets.zip -d $PWD/data
python -u $PWD/run_multi_dataset.py $1
