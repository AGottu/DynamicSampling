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
idealDevEM = {'drop': 0.53872, 'newsqa': 0.34245, 'squad2': 0.66015, 'quoref': 0.5089, 'ropes': 0.505, 'narrativeqa': 0.30622, 'squad': 0.3688, 'duorc': 0.2325}
idealDevF1 = {'drop': 0.572664, 'newsqa': 0.48785, 'squad2': 0.696, 'quoref': 0.558924, 'ropes': 0.5837, 'narrativeqa': 0.43874, 'squad': 0.5929, 'duorc': 0.30805}

@DataIterator.register("dynamic")
class DynamicIterator(BasicIterator):
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
        self.train_iterators = None

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
        assert trainDev == 'train'
        iterators = inst.fields["metadata"].metadata["iterators"]
        if self.train_iterators is None:
            self.train_iterators = {key:cycle(value) for (key,value) in iterators.items()} # Cycle for training iterators since we're sampling
        metrics = inst.fields["metadata"].metadata["val_metrics"]
        new_instances = []
        ########
        datasetNames = []
        sample_probs = []
        lossGaps = dict()
        EMGaps = dict()
        F1Gaps = dict()
        datasetNumbers = dict()
        if metrics is None:
            for datasetName, size in datasetSizes.items():
                datasetNames.append(datasetName)
                sample_probs.append(size)
        else:
            for datasetName, dev_metrics in metrics.items():
                dev_loss = dev_metrics['loss']
                dev_em = dev_metrics['em']
                dev_f1 = dev_metrics['f1']
                ideal_dev_loss = idealDevLosses[datasetName]
                ideal_dev_em = idealDevEM[datasetName]
                ideal_dev_f1 = idealDevF1[datasetName]
                loss_gap = max(0.01, dev_loss - ideal_dev_loss)
                em_gap = max(0.01, dev_em - ideal_dev_em)
                f1_gap = max(0.01, dev_f1 - ideal_dev_f1)
                lossGaps[datasetName] = loss_gap
                EMGaps[datasetName] = em_gap
                F1Gaps[datasetName] = f1_gap
                datasetNames.append(datasetName)
                sample_probs.append(em_gap)#(loss_gap)
            print('Loss Gaps: ', lossGaps)
            print('EM Gaps: ', EMGaps)
            print('F1 Gaps: ', F1Gaps)
            print('\n')
        tot = sum(sample_probs)
        sample_probs = [p/tot for p in sample_probs]
        for step in range(self._instances_per_epoch):
            datasetIndex = np.random.choice(len(sample_probs), p=sample_probs)
            datasetChosen = datasetNames[datasetIndex]
            if step % 3000 == 0:
                print('Step: ', step)
                print('%s Sampling Numbers: %s' % ('Size' if metrics is None else 'Dynamic', datasetNumbers))
            datasetNumbers[datasetChosen] = datasetNumbers.get(datasetChosen, 0) + 1
            inst = next(self.train_iterators[datasetChosen])
            assert inst is not None
            assert isinstance(inst, Instance)
            new_instances.append(inst)
        print('Final %s Sampling Numbers: %s' % ('Size' if metrics is None else 'Dynamic', datasetNumbers))
        assert len(datasetNumbers) == len(DATASETS)
        ########
        
        ### Ananth ###
        yield from super().__call__(new_instances, num_epochs, shuffle)