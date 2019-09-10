import json
import shutil
import sys
from torch import cuda

from allennlp.commands import main

# Use overrides to train on CPU.
overrides = json.dumps({"dataset_reader": {"allowed_datasets": "squad2"}, "trainer": {"cuda_device": [0,1,2,3] if cuda.is_available() else -1, "num_epochs": 20}})

config_file = '/mnt/c/Users/super/Documents/UCI/Bert-n-Pals-master/configs/nabert-plus-templated.json'
serialization_dir = '/mnt/c/Users/super/Documents/UCI/Bert-n-Pals-master/pals'

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package", "drop_bert",
    "-o", overrides
]

main()