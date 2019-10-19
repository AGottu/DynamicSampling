import logging
from overrides import overrides
from collections import deque
from typing import Dict, Union, Iterable, Iterator, List, Optional, Tuple, Deque
from collections import defaultdict
import itertools
from itertools import cycle
import numpy as np
import math
import random

import torch

from allennlp.common.registrable import Registrable
from allennlp.common.util import is_lazy, lazy_groups_of, ensure_list
from allennlp.data.dataset import Batch
from allennlp.data.fields import MetadataField
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
from allennlp.data.iterators.data_iterator import add_epoch_number
from allennlp.data.iterators import DataIterator
from allennlp.data.iterators import BasicIterator

logger = logging.getLogger(__name__)

TensorDict = Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]

DATASETS = ('drop', 'duorc', 'narrativeqa', 'newsqa', 'quoref', 'ropes', 'squad', 'squad2')
datasetSizes = {'drop': 77394, 'newsqa': 92543, 'squad2': 130310, 'quoref': 19392, 'ropes': 10302, 'narrativeqa': 32717, 'squad': 87596, 'duorc': 54746} # Approximate
devsetSizes = {'drop': 9529, 'duorc': 12224, 'narrativeqa': 3393, 'newsqa': 5154, 'quoref': 2407, 'ropes': 1194, 'squad': 10540, 'squad2': 11864}
idealDevLosses = {'drop': 1311376.45, 'newsqa': 3287434.267, 'squad2': 850474.152, 'quoref': 872098.2, 'ropes': 585288.3, 'narrativeqa': 2505892.7521, 'squad': 1133777.1, 'duorc': 3664924.6}

@DataIterator.register("multi")
class MultiIterator(BasicIterator):
    """
    An iterator that samples instances from 8 datasets based on individual dataset losses.
    """
    def __init__(
        self,
        batch_size: int = 32,
        instances_per_epoch: int = None,
        max_instances_in_memory: int = None,
        cache_instances: bool = False,
        track_epoch: bool = False,
        maximum_samples_per_batch: Tuple[str, int] = None,
    ) -> None:
        super().__init__(batch_size, instances_per_epoch, max_instances_in_memory, cache_instances, track_epoch, maximum_samples_per_batch)
        self.dev_iterators = None
        self.dataset_choice = 'all'

    def chooseDataset(self, datasetName: str):
        self.dataset_choice = datasetName
        print('Dataset Chosen: %s' % datasetName.upper())

    @overrides
    def __call__(
        self, instances: Iterable[Instance], num_epochs: int = None, shuffle: bool = True
    ) -> Iterator[TensorDict]:

        ### Ananth ###
        assert num_epochs == 1 # Iterator should only be called from train_epoch or validation_loss, both of which are for 1 epoch
        instances = list(instances)
        assert len(instances) == 1
        inst = instances[0]
        trainDev = inst.fields["metadata"].metadata["trainDev"]
        assert trainDev == 'dev'
        iterators = inst.fields["metadata"].metadata["iterators"]
        if self.dev_iterators is None:
            self.dev_iterators = {key:list(value) for (key,value) in iterators.items()}
        new_instances = self.dev_iterators[self.dataset_choice] if self.dataset_choice in DATASETS else self.dev_iterators['ropes']
        ### Ananth ###
        yield from super().__call__(new_instances, num_epochs, shuffle)