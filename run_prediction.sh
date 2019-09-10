#!/bin/sh
pip install gdown
python download_files.py
unzip -o data/all_datasets.zip -d data
python run_multi_dataset.py
