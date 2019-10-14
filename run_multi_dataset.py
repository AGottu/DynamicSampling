import os
import json
import shutil
import sys
from torch import cuda

from allennlp.commands import main

assert len(sys.argv) == 4
# Use overrides to train on CPU.
dynamic = sys.argv[1] == 'dynamic'
instancesPerEpoch = 50000 if dynamic else None
iteratorType = 'dynamic' if dynamic else 'basic'
numEpochs = int(sys.argv[3])
overrides_dict = {"dataset_reader": {"lazy": True, "allowed_datasets": sys.argv[1], "instances_per_epoch": instancesPerEpoch}, "iterator": {"type": iteratorType, "instances_per_epoch": instancesPerEpoch}, "trainer": {"cuda_device": 0 if cuda.is_available() else -1, "num_epochs": numEpochs}}
if dynamic:
    overrides_dict["trainer"]["type"] = 'dynamic-sample'
overrides = json.dumps(overrides_dict)

config_file = '%s/configs/nabert-plus-templated.json' % os.getcwd()
serialization_dir = sys.argv[2]

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