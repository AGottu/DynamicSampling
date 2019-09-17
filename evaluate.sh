#!/bin/sh
pip install gdown
python $PWD/download_files.py
unzip -o $PWD/data/all_datasets.zip -d $PWD/data
python $PWD/download_drop.py
mkdir /agottumu/results
python -u $PWD/evaluate_multi_dataset.py drop $1 $2
python -u $PWD/evaluate_multi_dataset.py duorc $1 $2
python -u $PWD/evaluate_multi_dataset.py narrativeqa $1 $2
python -u $PWD/evaluate_multi_dataset.py newsqa $1 $2
python -u $PWD/evaluate_multi_dataset.py quoref $1 $2
python -u $PWD/evaluate_multi_dataset.py ropes $1 $2
python -u $PWD/evaluate_multi_dataset.py squad $1 $2
python -u $PWD/evaluate_multi_dataset.py squad2 $1 $2
python -u $PWD/evaluate_multi_dataset.py all $1 $2