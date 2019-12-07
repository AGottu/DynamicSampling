#!/bin/sh
pip install git+https://github.com/allenai/allennlp.git
pip install gdown
python $PWD/download_files.py
unzip -o $PWD/data/all_datasets.zip -d $PWD/data
#python $PWD/download_drop.py
rm -r /agottumu/results
mkdir /agottumu/results
CUDA_VISIBLE_DEVICES=0,1 python -u $PWD/run_multi_dataset.py $1 $2 $3
