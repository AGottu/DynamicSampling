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
datasetSizes = {'drop': 77394, 'newsqa': 92543, 'squad2': 130310, 'quoref': 19392, 'ropes': 10924, 'narrativeqa': 32717, 'squad': 87596, 'duorc': 54746} # Approximate
reducedSizes = {'drop': 69990, 'newsqa': 61658, 'squad2': 115168, 'quoref': 17600, 'ropes': 10565, 'narrativeqa': 24336, 'squad': 69005, 'duorc': 35315}
devsetSizes = {'drop': 9530, 'duorc': 12224, 'narrativeqa': 3393, 'newsqa': 5154, 'quoref': 2407, 'ropes': 1688, 'squad': 10570, 'squad2': 11864}

idealDevLosses = {'drop': 1311375.6, 'newsqa': 3287434.3085, 'squad2': 850474.5231, 'quoref': 872099.2, 'ropes': 130333.73, 'narrativeqa': 2505894.0, 'squad': 1993947.91, 'duorc': 3664924.6}

idealDevEM = {'drop': 0.54407, 'newsqa': 0.3533, 'squad2': 0.641436, 'quoref': 0.525135, 'ropes': 0.67535545, 'narrativeqa': 0.31506, 'squad': 0.574456, 'duorc': 0.2325}
idealDevF1 = {'drop': 0.58, 'newsqa': 0.49783, 'squad2': 0.6766, 'quoref': 0.5781, 'ropes': 0.72106, 'narrativeqa': 0.4428, 'squad': 0.7351353, 'duorc': 0.30805}
# Squad 2 Old: 0.66015, 0.696
#idealDevEM = {'drop': 0.5341, 'newsqa': 0.346527, 'squad2': 0.6481, 'quoref': 0.53012, 'ropes': 0.51088777, 'narrativeqa': 0.308576481, 'squad': 0.5666, 'duorc': 0.2307}
#idealDevF1 = {'drop': 0.5689, 'newsqa': 0.4862534, 'squad2': 0.6825, 'quoref': 0.586261, 'ropes': 0.5875, 'narrativeqa': 0.4377, 'squad': 0.72442668, 'duorc': 0.3078575}

def cumulativeMetrics(metrics):
    cumulativeEM = 0.0
    cumulativeF1 = 0.0
    cumulativeSize = 0
    for datasetName, dev_metrics in metrics.items():
        dev_loss = dev_metrics['loss']
        dev_em = dev_metrics['em']
        dev_f1 = dev_metrics['f1']
        dev_size = devsetSizes[datasetName]
        cumulativeEM += (dev_em * dev_size)
        cumulativeF1 += (dev_f1 * dev_size)
        cumulativeSize += dev_size
    cumulativeEM /= cumulativeSize
    cumulativeF1 /= cumulativeSize
    print('Cumulative EM: ', cumulativeEM)
    print('Cumulative F1: ', cumulativeF1)

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
        self.roundRobinIndex = 0

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
        
        sampling_method = inst.fields["metadata"].metadata["sampling_method"]
        scheduling = inst.fields["metadata"].metadata["scheduling"]
        dynamic_metric = inst.fields["metadata"].metadata["dynamic_metric"]
        print('Sampling Method: ', sampling_method)
        print('Scheduling: ', scheduling)
        print('Metric: ', dynamic_metric)
        
        new_instances = []
        ########
        datasetNames = []
        sample_probs = []
        lossGaps = dict()
        EMGaps = dict()
        F1Gaps = dict()
        datasetNumbers = dict()
        if metrics is None or sampling_method == 'size':
            for datasetName, size in datasetSizes.items():
                datasetNames.append(datasetName)
                sample_probs.append(size)
        elif sampling_method == 'uniform':
            for datasetName, size in datasetSizes.items():
                datasetNames.append(datasetName)
                sample_probs.append(1.0 / len(datasetSizes))
        else:
            assert sampling_method == 'dynamic'
            for datasetName, dev_metrics in metrics.items():
                dev_loss = dev_metrics['loss']
                dev_em = dev_metrics['em']
                dev_f1 = dev_metrics['f1']
                ideal_dev_loss = idealDevLosses[datasetName]
                ideal_dev_em = idealDevEM[datasetName]
                ideal_dev_f1 = idealDevF1[datasetName]
                loss_gap = max(0.01, dev_loss - ideal_dev_loss)
                em_gap = max(0.01, ideal_dev_em - dev_em)
                f1_gap = max(0.01, ideal_dev_f1 - dev_f1)
                lossGaps[datasetName] = loss_gap
                EMGaps[datasetName] = em_gap
                F1Gaps[datasetName] = f1_gap
                datasetNames.append(datasetName)
                if dynamic_metric == 'em':
                    sample_probs.append(em_gap)
                elif dynamic_metric == 'f1':
                    sample_probs.append(f1_gap)
                elif dynamic_metric == 'em+f1':
                    sample_probs.append(em_gap + f1_gap)
                else:
                    assert dynamic_metric == 'loss' 
                    sample_probs.append(loss_gap)
            print('Loss Gaps: ', lossGaps)
            print('EM Gaps: ', EMGaps)
            print('F1 Gaps: ', F1Gaps)
            print('\n')
            print(metrics)
            print('\n')
            cumulativeMetrics(metrics)
            print('\n')
        tot = sum(sample_probs)
        sample_probs = [p/tot for p in sample_probs]
        if scheduling == 'rr':
            datasetChosen = datasetNames[self.roundRobinIndex]
            for step in reducedSizes[datasetChosen]:
                if step % 3000 == 0:
                    print('Step: ', step)
                    print('Round Robin Numbers: %s' % datasetNumbers)
                datasetNumbers[datasetChosen] = datasetNumbers.get(datasetChosen, 0) + 1
                inst = next(self.train_iterators[datasetChosen])
                assert inst is not None
                assert isinstance(inst, Instance)
                new_instances.append(inst)
            self.roundRobinIndex += 1
            self.roundRobinIndex %= len(datasetNames)
        else:
            assert scheduling in ('mixed_unmixed', 'mixed_mixed')
            datasetChosen = None
            for step in range(self._instances_per_epoch):
                if scheduling == 'mixed_mixed' or (step % self._batch_size == 0):
                    datasetIndex = np.random.choice(len(sample_probs), p=sample_probs)
                    datasetChosen = datasetNames[datasetIndex]
                if step % 3000 == 0:
                    print('Step: ', step)
                    print('%s Sampling Numbers: %s' % (sampling_method, datasetNumbers))
                datasetNumbers[datasetChosen] = datasetNumbers.get(datasetChosen, 0) + 1
                inst = next(self.train_iterators[datasetChosen])
                assert inst is not None
                assert isinstance(inst, Instance)
                new_instances.append(inst)
        print('Final %s Sampling Numbers: %s' % (sampling_method, datasetNumbers))
        assert len(datasetNumbers) == len(DATASETS)
        ########
        
        ### Ananth ###
        yield from super().__call__(new_instances, num_epochs, shuffle)