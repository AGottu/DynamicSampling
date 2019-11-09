import os
import json
import shutil
import sys
from torch import cuda

from allennlp.commands import main

dataset = sys.argv[1]
overrides = json.dumps({"dataset_reader": {"lazy": True, "allowed_datasets": dataset}, "iterator": {"type": 'basic', "batch_size": 4}, "validation_iterator": {"type": 'basic', "batch_size": 4}})

archive_file = sys.argv[2]
input_file = '%s/data/all_datasets/dev' % os.getcwd()
output_file = '%s/%s.json' % (sys.argv[3], dataset)
weights_file = '/agottumu/dynamic/model_state_epoch_12.th'

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    archive_file,
    input_file,
    "--output-file", output_file,
    "--cuda-device", 0,
    "--include-package", "drop_bert",
    "-o", overrides,
    "--weights-file", weights_file
]

main()