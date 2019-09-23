import os
import json
import shutil
import sys
from torch import cuda

from allennlp.commands import main

# Use overrides to train on CPU.
instancesPerEpoch = 50000 if sys.argv[1] == 'all-sample' else None
numEpochs = 5
overrides = json.dumps({"dataset_reader": {"lazy": True, "allowed_datasets": sys.argv[1], "instances_per_epoch": instancesPerEpoch, "numEpochs": numEpochs}, "iterator": {"instances_per_epoch": instancesPerEpoch}, "trainer": {"cuda_device": 0 if cuda.is_available() else -1, "num_epochs": numEpochs}})

model_archive = '%s/pals/model.tar.gz' % os.getcwd() if len(sys.argv) < 3 else sys.argv[2]
config_file = '%s/configs/nabert-plus-templated.json' % os.getcwd()
serialization_dir = '%s/palsFineTune' % os.getcwd() if len(sys.argv) < 4 else sys.argv[3]

# Training will fail if the serialization directory already
# has stuff in it. If you are running the same training loop
# over and over again for debugging purposes, it will.
# Hence we wipe it out in advance.
# BE VERY CAREFUL NOT TO DO THIS FOR ACTUAL TRAINING!
shutil.rmtree(serialization_dir, ignore_errors=True)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "fine-tune",
    "-m", model_archive,
    "-c", config_file,
    "-s", serialization_dir,
    "--include-package", "drop_bert",
    "-o", overrides
]

main()