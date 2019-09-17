import os
import json
import shutil
import sys
from torch import cuda

from allennlp.commands import main

dataset = sys.argv[1]
overrides = json.dumps({"dataset_reader": {"lazy": True, "allowed_datasets": dataset}})

archive_file = sys.argv[2]
input_file = '%s/data/all_datasets/dev' % os.getcwd()
output_file = '%s/%s.json' % (sys.argv[3], dataset)

# Assemble the command into sys.argv
sys.argv = [
    "allennlp",  # command name, not used by main
    "evaluate",
    archive_file,
    input_file,
    "--output-file", output_file,
    "--include-package", "drop_bert",
    "-o", overrides
]

main()