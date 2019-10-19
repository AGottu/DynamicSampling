import logging
from overrides import overrides
import math
import os
import time
import datetime
import traceback
from typing import Dict, Optional, List, Tuple, Union, Iterable, Any

import torch
import torch.optim.lr_scheduler

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError, parse_cuda_device
from allennlp.common.util import dump_metrics, gpu_memory_mb, peak_memory_mb, lazy_groups_of
from allennlp.common.tqdm import Tqdm
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator, TensorDict
from allennlp.data.fields import MetadataField
from allennlp.models.model import Model
from allennlp.nn import util as nn_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorboardWriter
from allennlp.training.trainer_base import TrainerBase
from allennlp.training import util as training_util
from allennlp.training.moving_average import MovingAverage
from allennlp.training.trainer import Trainer
from allennlp.training.trainer_pieces import TrainerPieces

logger = logging.getLogger(__name__)
DATASETS = ('drop', 'duorc', 'narrativeqa', 'newsqa', 'quoref', 'ropes', 'squad', 'squad2')
devsetSizes = {'drop': 9529, 'duorc': 12224, 'narrativeqa': 3393, 'newsqa': 5154, 'quoref': 2407, 'ropes': 1194, 'squad': 10540, 'squad2': 11864}

@TrainerBase.register("dynamic-sample")
class DynamicTrainer(Trainer):
    def getValidationMetrics(self) -> Dict[str, float]:
        validationMetrics = dict()
        for datasetName in DATASETS:
            self._validation_iterator.chooseDataset(datasetName)
            val_loss, num_batches = super()._validation_loss()
            val_metrics = training_util.get_metrics(self.model, val_loss, num_batches, reset=True)
            validationMetrics[datasetName] = val_metrics
            print('\nCalculated Validation Metrics for %s Dataset' % datasetName.upper())
        self.iterator.chooseDataset('all') # Revert dataset choice to 'all' once Dataset Losses are calculated
        return validationMetrics

    @overrides
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        if epoch > 0:
            val_metrics = self.getValidationMetrics()
            print('\nCalculated Validation Metrics for Epoch %d' % epoch)
        else:
            val_metrics = None
        self.train_data = list(self.train_data)
        assert len(self.train_data) == 1
        self.train_data[0].fields["metadata"].metadata["val_metrics"] = val_metrics
        self.train_data[0].fields["metadata"].metadata["epoch"] = epoch
        return super()._train_epoch(epoch)

    # Requires custom from_params.
    @classmethod
    def from_params(  # type: ignore
        cls,
        params: Params,
        serialization_dir: str,
        recover: bool = False,
        cache_directory: str = None,
        cache_prefix: str = None
    ) -> "DynamicTrainer":

        pieces = TrainerPieces.from_params(params, serialization_dir, recover, cache_directory, cache_prefix)
        model = pieces.model
        iterator = pieces.iterator
        train_data = pieces.train_dataset
        validation_data = pieces.validation_dataset
        validation_iterator = pieces.validation_iterator
        params = pieces.params

        patience = params.pop_int("patience", None)
        validation_metric = params.pop("validation_metric", "-loss")
        shuffle = params.pop_bool("shuffle", True)
        num_epochs = params.pop_int("num_epochs", 20)
        cuda_device = parse_cuda_device(params.pop("cuda_device", -1))
        grad_norm = params.pop_float("grad_norm", None)
        grad_clipping = params.pop_float("grad_clipping", None)
        lr_scheduler_params = params.pop("learning_rate_scheduler", None)
        momentum_scheduler_params = params.pop("momentum_scheduler", None)

        if isinstance(cuda_device, list):
            model_device = cuda_device[0]
        else:
            model_device = cuda_device
        if model_device >= 0:
            # Moving model to GPU here so that the optimizer state gets constructed on
            # the right device.
            model = model.cuda(model_device)

        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        opt = params.pop("optimizer")
        optimizer = Optimizer.from_params(parameters, opt)
        if "moving_average" in params:
            moving_average = MovingAverage.from_params(
                params.pop("moving_average"), parameters=parameters
            )
        else:
            moving_average = None

        if lr_scheduler_params:
            lr_scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params)
        else:
            lr_scheduler = None
        if momentum_scheduler_params:
            momentum_scheduler = MomentumScheduler.from_params(optimizer, momentum_scheduler_params)
        else:
            momentum_scheduler = None

        if "checkpointer" in params:
            if (
                "keep_serialized_model_every_num_seconds" in params
                or "num_serialized_models_to_keep" in params
            ):
                raise ConfigurationError(
                    "Checkpointer may be initialized either from the 'checkpointer' key or from the "
                    "keys 'num_serialized_models_to_keep' and 'keep_serialized_model_every_num_seconds'"
                    " but the passed config uses both methods."
                )
            checkpointer = Checkpointer.from_params(params.pop("checkpointer"))
        else:
            num_serialized_models_to_keep = params.pop_int("num_serialized_models_to_keep", 20)
            keep_serialized_model_every_num_seconds = params.pop_int(
                "keep_serialized_model_every_num_seconds", None
            )
            checkpointer = Checkpointer(
                serialization_dir=serialization_dir,
                num_serialized_models_to_keep=num_serialized_models_to_keep,
                keep_serialized_model_every_num_seconds=keep_serialized_model_every_num_seconds,
            )
        model_save_interval = params.pop_float("model_save_interval", None)
        summary_interval = params.pop_int("summary_interval", 100)
        histogram_interval = params.pop_int("histogram_interval", None)
        should_log_parameter_statistics = params.pop_bool("should_log_parameter_statistics", True)
        should_log_learning_rate = params.pop_bool("should_log_learning_rate", False)
        log_batch_size_period = params.pop_int("log_batch_size_period", None)

        params.assert_empty(cls.__name__)
        return cls(
            model,
            optimizer,
            iterator,
            train_data,
            validation_data,
            patience=patience,
            validation_metric=validation_metric,
            validation_iterator=validation_iterator,
            shuffle=shuffle,
            num_epochs=num_epochs,
            serialization_dir=serialization_dir,
            cuda_device=cuda_device,
            grad_norm=grad_norm,
            grad_clipping=grad_clipping,
            learning_rate_scheduler=lr_scheduler,
            momentum_scheduler=momentum_scheduler,
            checkpointer=checkpointer,
            model_save_interval=model_save_interval,
            summary_interval=summary_interval,
            histogram_interval=histogram_interval,
            should_log_parameter_statistics=should_log_parameter_statistics,
            should_log_learning_rate=should_log_learning_rate,
            log_batch_size_period=log_batch_size_period,
            moving_average=moving_average,
        )