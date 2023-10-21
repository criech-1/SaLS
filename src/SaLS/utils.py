"""Provides utility functions."""
import os
import sys
import logging
import random
import numpy as np
import torch

from abc import ABC, abstractmethod
from copy import deepcopy
from math import sqrt, log, pi
from typing import List, Dict, Union, Optional, Callable, Tuple, Any, Iterable

from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

def seed_all_rng(seed: Union[int, None] = None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed: The seed value to use. If None, will use a strong random seed.
    """
    if seed is None:
        seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    return seed

def set_seed(local_seed=None):
    """Sets the global random seed.

    If ``local_seed`` is None, a random seed is chosen.

    Args:
        local_seed: The seed value to use
    """
    global seed
    seed = seed_all_rng(local_seed)
    print(f'random seed: {seed}')

def accuracy(input: Tensor,
             target: Tensor,
             reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
             classification: bool = True) -> Tensor:
    """Computes the accuracy.

    This method computes the argmax over the dimension 1 of the
    input and compares this prediction with the target.
    In addition, the result can be reduced with the reduction_fn

    Args:
        input: (shape [B, C]) The class input
        target: (shape [B]) The target values
        reduction_fn: A reduction function that can be applied to shape [B]

    Returns:
        A tensor containing the reduced accuracy value
    """
    if not classification:
        return torch.zeros([]).to(input)
    if reduction_fn is None:
        reduction_fn = torch.mean
    return 100.0 * reduction_fn(torch.argmax(input, dim=1) == target)

def squared_error(input: Tensor,
                  target: Tensor,
                  reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
                  classification: bool = True) -> Tensor:
    """Computes the squared error.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B, C]) The target values

    Returns:
        A tensor containing the squared error
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    if classification:
        target = torch.eye(input.shape[1], device=input.device, dtype=input.dtype)[target]
    return reduction_fn(torch.mean((input - target) ** 2, dim=-1))


def absolute_error(input: Tensor,
                   target: Tensor,
                   reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
                   classification: bool = True) -> Tensor:
    """Computes the absolute error.

    Args:
        input: (shape [B, C]) The input
        target: (shape [B, C]) The target values

    Returns:
        A tensor containing the absolute error
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    if classification:
        target = torch.eye(input.shape[1], device=input.device, dtype=input.dtype)[target]
    return reduction_fn(torch.mean(torch.abs(input - target), dim=-1))


def standard_deviation(input: Tensor,
                       target: Tensor,
                       reduction_fn: Optional[Callable[[Tensor], Tensor]] = None,
                       classification: bool = True) -> Tensor:
    """Computes the standard deviation.

    Args:
        input: (shape [B, N, C]) The input
        target: The target values

    Returns:
        A tensor containing the standard deviation
    """
    if reduction_fn is None:
        reduction_fn = torch.mean
    return reduction_fn(torch.std(input, dim=1))


class Metric(ABC):
    """Base class for all Metrics.

    The metric should implement ``update`` and ``summarize``.
    This allows to compute the metrics in an online setting.
    """

    assert_mean = True

    @abstractmethod
    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of class probabilities and
        target.

        Args:
            input: The class probabilities as a list of (shape [B, C]) tensors
            target: (shape [B]) The target values
        """
        raise NotImplementedError

    @abstractmethod
    def summarize(self):
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        raise NotImplementedError


class FunctionalMetric(Metric):
    """A functional metric.

    This metric applies a given function on an index of the probabilities and
    averages the resulting value over the whole dataset.

    With ``reduction = 'mean'``, this computes the entropy for each element of
        the ``input`` list.
    With ``reduction = None``, the sample-wise entropy is computed for each
        element of the ``input`` list.
    With ``reduction = 'correct vs. incorrect'``, the mean entropy is computed
        over the correct and incorrect samples, respectively.
    """

    def __init__(self,
                 function: Callable[[Tensor,
                                     Tensor,
                                     Callable[[Tensor], Tensor],
                                     bool],
                                    Tensor],
                 len_data: int,
                 dim: int = -1,
                 reduction: Optional[str] = 'mean',
                 is_classification: bool = True,
                 assert_mean: bool = True):
        """FunctionalMetric initializer.

        Args:
            function: A function that maps the input and target tensor to
                a single scalar reduced to the sum
            len_data: The length of the dataset
            dim: The index of the input that should be used to compute
                the metric
        """
        super().__init__()
        self.function = function
        self.len_data = len_data
        self.value = [] if reduction is None else 0.
        self.dim = dim
        self.is_classification = is_classification
        self.reduction = reduction
        self.assert_mean = assert_mean

        if reduction == 'mean':
            self.reduction_fn: Callable[[Tensor], Tensor] = torch.sum
        elif reduction == 'correct vs. incorrect':
            self.reduction_fn: Callable[[Tensor], Tensor] = lambda x, **kwargs: x
            self.n_correct = 0
        elif reduction is None:
            self.reduction_fn: Callable[[Tensor], Tensor] = lambda x, **kwargs: x.tolist()
        else:
            raise ValueError(f'Unknown reduction: {reduction}')

    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of input and target.

        Args:
            input: The class probabilities as a list of tensors
                 each shape [B, C] if self.assert_mean else of shape [B, N, C]
            target: (shape [B]) The target values
        """
        input_dim = input[self.dim]
        new_value = self.function(input_dim, target,
                                  reduction_fn=self.reduction_fn,
                                  classification=self.is_classification)
        if self.reduction == 'correct vs. incorrect':
            in_mean = input_dim if input_dim.ndim == 2 else input_dim.mean(dim=1)
            correct_inds: Tensor = torch.argmax(in_mean, dim=-1) == target
            self.n_correct += torch.sum(correct_inds).item()

            correct = new_value[correct_inds].sum()
            incorrect = new_value[~correct_inds].sum()

            new_value = torch.stack([correct, incorrect])
        self.value = self.value + new_value

    def summarize(self) -> Union[float, List[float], Dict[str, float]]:
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        if self.reduction == 'mean':
            return self.value.item() / self.len_data
        elif self.reduction == 'correct vs. incorrect':
            return {'correct': np.divide(self.value[0].item(), self.n_correct),
                    'incorrect': np.divide(self.value[1].item(), self.len_data - self.n_correct)}
        elif self.reduction is None:
            return self.value


class ApplyFunctionalMetric(Metric):
    """The metric that applies the FunctionalMetric the each element of the
    input list."""

    def __init__(self,
                 function: Callable[[Tensor,
                                     Tensor,
                                     Callable[[Tensor], Tensor]], Tensor],
                 len_data: int,
                 reduction: Optional[str] = 'mean',
                 is_classification: List[bool] = None,
                 assert_mean: bool = True):
        """ApplyFunctionalMetric initializer.

        Args:
            function: A function that maps the input and target tensor to
                a single scalar reduced to the sum
            len_data: The length of the dataset
            is_classification: A list of boolean values indicating whether the metric is
                a classification metric or a regression metric
        """
        super().__init__()
        self.assert_mean = assert_mean
        self.functional_metrics = [FunctionalMetric(function, len_data, i, reduction, is_classification[i])
                                   for i in range(len(is_classification))]

    def update(self, input: List[Tensor], target: Tensor):
        """Updates the metrics with a new batch of class probabilities and
        target.

        Args:
            input: The input as a list of (shape [B, C])
                tensors
            target: (shape [B]) The target values
        """
        for f in self.functional_metrics:
            f.update(input, target)

    def summarize(self) -> List[float]:
        """Summarizes the metric value after all batches were observed.

        Returns:
            The value of the metric aggregated over all batches
        """
        return [f.summarize() for f in self.functional_metrics]


class StandardDeviation(ApplyFunctionalMetric):
    assert_mean = False

    def __init__(self,
                 len_data: int,
                 reduction: Optional[str] = 'mean',
                 is_classification: List[bool] = None):
        """Standard deviation initializer.

        Args:
            len_data: The length of the dataset
            reduction: The reduction to apply to the metric value. Possible
                reductions: mean, None
            is_classification: A list of booleans indicating whether the metric
                is a classification metric or a regression metric
        """
        super().__init__(standard_deviation, len_data, reduction, is_classification)


class MetricSet:
    """Set of different metrics."""

    all_metrics = {'accuracy', 'brier score', 'negative log likelihood',
                   'calibration', 'entropy', 'entropy histogram',
                   'entropy correct vs. incorrect', 'mse', 'mae', 'std',
                   'std histogram', 'std correct vs. incorrect'}

    @staticmethod
    def metric_to_class(metric: str,
                        len_data: int,
                        dim: int,
                        is_classification: List[bool]) -> Metric:
        """Converts a metric name to a metric class.

        Args:
            metric: The name of the metric

        Returns:
            The metric class
        """
        dim_classification = is_classification[dim]
        if metric == 'accuracy':
            return FunctionalMetric(accuracy, len_data, dim, 'mean', dim_classification)
        elif metric == 'mse':
            return FunctionalMetric(squared_error, len_data, dim, 'mean', dim_classification)
        elif metric == 'mae':
            return FunctionalMetric(absolute_error, len_data, dim, 'mean', dim_classification)
        elif metric == 'std':
            return StandardDeviation(len_data, 'mean', is_classification)
        elif metric == 'std histogram':
            return StandardDeviation(len_data, None, is_classification)
        elif metric == 'std correct vs. incorrect':
            return StandardDeviation(len_data, 'correct vs. incorrect', is_classification)

    def __init__(self,
                 len_data: int,
                 metrics: Union[str, Iterable[str]] = None,
                 dim: int = -1,
                 is_classification: List[bool] = None):
        """MetricSet initializer.

        Args:
            len_data: The length of the dataset
            metrics: Either 'all' or a list of metrics strings
                (Possible values: 'accuracy', 'brier score',
                'negative log likelihood', 'entropy', 'calibration',
                'entropy histogram')
            dim: The index of the probabilities that should be used to compute
                the metric (except entropy which is computed over the full list)
        """
        if metrics == 'all':
            metrics = self.all_metrics
        elif metrics is None:
            metrics = set()
        elif isinstance(metrics, str):
            metrics = {metrics}
        elif not isinstance(metrics, set):
            metrics = set(metrics)
        metrics = list(metrics & self.all_metrics)
        self.metric_classes = {metric: self.metric_to_class(metric, len_data, dim, is_classification)
                               for metric in metrics}
        self.is_classification = is_classification

    def update(self, logits: Union[List[Tensor], Tensor], target: Tensor):
        """Updates the metrics with a new batch of class logits and
        target.

        Args:
            logits: The class logits as a list of (shape [B, C]) tensors or a
                single (shape [B, C]) tensor
            target: (shape [B]) The target values
        """
        if self.metric_classes:
            if not isinstance(logits, list):
                logits = [logits]

            for logit in logits:
                if not isinstance(logit, Tensor):
                    raise TypeError(f'Expected logits to be a list of Tensors, '
                                    f'got {type(logit)}')
                if logit.ndim == 2:
                    logit.unsqueeze_(1)

            input = [F.softmax(logit, dim=-1) if is_classification else logit
                     for is_classification, logit in zip(self.is_classification, logits)]

            mean = [i.mean(dim=1) for i in input]

            for metric_class in self.metric_classes.values():
                if metric_class.assert_mean:
                    metric_class.update(mean, target)
                else:
                    metric_class.update(input, target)

    def summarize(self) -> Dict[str, Any]:
        """Summarizes the metrics values after all batches were observed.

        Returns:
            A dict containing the summerized values
        """
        return {metric: metric_class.summarize()
                for metric, metric_class in self.metric_classes.items()}


def run_epoch(model: nn.Module,
              dataloader: DataLoader,
              criterion: Callable[[Tensor, Tensor], Tensor],
              is_classification: Union[bool, List[bool]] = True,
              optimizer: Optional[torch.optim.Optimizer] = None,
              train: bool = True,
              metrics: Union[str, List[str]] = None,
              metrics_dim: int = -1,
              prefix: str = '',
              return_if_loss_nan_or_inf: bool = False,
              device: str = None,
              logit_values: bool = False) \
        -> Dict[str, Union[float, np.array]]:
    """Runs one epoch over the dataloader.

    If ``train`` is True, the model is optimized, else the model is evaluated.

    Args:
        model: A PyTorch model
        dataloader: A PyTorch dataloader
        criterion: A function mapping the logits and targets to the loss
        optimizer: A PyTorch optimizer
        train: Whether the model is trained or evaluated
        metrics: Either 'all' or a list of metrics strings
            (Possible values: 'accuracy', 'brier score',
            'negative log likelihood', 'entropy', 'calibration',
            'entropy histogram')
        metrics_dim: The index of the logits that should be used to compute the
            loss
        prefix: The description of the tqdm bar
        return_if_loss_nan: Whether to return if the loss is NaN

    Returns:
        The metrics over the epoch
    """
    model.train() if train else model.eval()

    if not isinstance(is_classification, list):
        is_classification = [is_classification]

    metrics_obj = MetricSet(len(dataloader.dataset), metrics, metrics_dim,
                            is_classification)

    mean_loss = torch.as_tensor(0., device=device)
    
    if logit_values:
        logits_val = []

    model.to(device)
    with tqdm(dataloader, file=sys.stdout) as t:
        t.set_description(prefix)
        for step, (features, targets) in enumerate(t):
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(features)
            logit = logits[metrics_dim]

            maxlogit, _ = torch.max(logit, dim=1)

            if logit_values:
                logits_val.append(maxlogit.detach().cpu().numpy()[0])
            
            loss = criterion(logit, targets.float())

            if loss.isnan() or loss.isinf():
                if return_if_loss_nan_or_inf:
                    return {'loss': loss.item()}
                elif loss.isnan():
                    print(f'Loss is nan in step {step}. This step is ignored for training.')
            if not loss.isnan() and train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            mean_loss += loss.detach() / len(dataloader)
            metrics_obj.update([logit.detach() for logit in logits], targets.detach())
            t.set_postfix(loss=loss.detach().item())

    metrics_dict = metrics_obj.summarize()
    metrics_dict['loss'] = mean_loss.item()
    
    if logit_values:
        metrics_dict['logits'] = logits_val

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics_dict


def get_verbose_string(d):
    """Creates the string that summarizes the dictionary of metrics.

    Args:
        d: The dictionary of metrics

    Returns:
        A string containing the metrics
    """
    assert 'loss' in d.keys()
    verbose_string = f'loss: {d["loss"]: .5f}'
    for key, value in d.items():
        if key != 'loss' and key != 'logits':
            verbose_string += f', {key}: {value:.2f}'
            if key == 'accuracy':
                verbose_string += '%'
    return verbose_string


def get_dataset_and_name(dataloader: DataLoader):
    """Returns the dataset of the dataloader and its name.

    Args:
        dataloader: A PyTorch dataloader

    Returns:
        The dataset and the name
    """
    dataset = dataloader.dataset
    ds = dataset
    while isinstance(ds, Subset):
        ds = ds.dataset
    return dataset, type(ds).__name__


def fit_pnn(model: nn.Module, 
            dataloader_train: DataLoader,
            parameters: Dict,
            device: str,
            name: str,
            dataloader_test: Optional[DataLoader] = None,
            dataloader_val: DataLoader = None,
            retrain: Optional[bool] = False) \
                -> nn.Module:

    print('Parameters: ', parameters)
    learning_rate = parameters['learning_rate']
    weight_decay = parameters['weight_decay']
    patience = parameters['patience']
    num_epochs = parameters['num_epochs']

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()
    num_classes = len(dataloader_train.dataset.classes)
    output_size = 1 #num_classes if num_classes > 1 else 2

    metrics = []

    if name is not None:
        model_path = os.path.join('models', name)

        if not os.path.exists(os.path.join('models', name)):
            os.makedirs(model_path)

    if len(model.networks_labels) > 0:
        if name is not None:
            torch.save(model.full_state_dict(), os.path.join(model_path, 'full_state_dict_{}.pt'.format(len(model.networks_labels))))
    else:
        if name is not None and not os.path.exists(os.path.join(model_path, 'full_state_dict_-1.pt')):
            torch.save(model.full_state_dict(), os.path.join(model_path, 'full_state_dict_-1.pt'))
    
    previous_tasks = model.previous_tasks  # + 1 for prior task # model.previous_tasks -> returns len(self.networks) + 1

    task = previous_tasks 
    torch.cuda.empty_cache()
    print(f'Task {task}: ' + str(dataloader_train.dataset.classes))

    is_classification = True
    model.add_new_column(is_classification=is_classification, output_size=output_size)

    state_copy = deepcopy(model.state_dict())

    training_done = False
    while not training_done:
        metrics_run_epoch = None

        if metrics_run_epoch is None:
            metrics_run_epoch = ['accuracy']

            # variables for early stopping
            best_model_state_dict = model.state_dict()
            metric_for_best_model = 'loss'
            metric_should_be_large = False
            early_stopping_coefficient = 1 if metric_should_be_large else -1
            best_value = -float('Inf') * early_stopping_coefficient
            steps_without_improvement = 0

            metrics = {'train': [], 'val': []} if dataloader_val is not None else {'train': []}

            if dataloader_test is not None:
                metrics['test'] = []

            criterion.to(device)
            parameter_container = criterion if hasattr(criterion, 'model') else model
            params = [p for p in parameter_container.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
            lr_schedulers = [
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, verbose=True, patience=5),
            ]

            for epoch in range(num_epochs):
                string_length = str(len(str(num_epochs - 1)))
                prefix = ('Epoch {epoch:' + string_length + 'd}: ').format(epoch=epoch + 1)
                train_metrics = run_epoch(model=model,
                                          dataloader=dataloader_train,
                                          criterion=criterion,
                                          is_classification=is_classification,
                                          optimizer=optimizer,
                                          train=True,
                                          metrics=metrics_run_epoch,
                                          prefix=prefix + 'Train',
                                          device=device)
                metrics['train'].append(train_metrics)

                verbose_string = prefix
                verbose_string += f'Train: {get_verbose_string(train_metrics)} | '

                if dataloader_val is not None:
                    val_metrics = run_epoch(model=model,
                                            dataloader=dataloader_val,
                                            criterion=criterion,
                                            is_classification=is_classification,
                                            train=False,
                                            metrics=metrics_run_epoch,
                                            prefix=prefix + 'Val',
                                            device=device,
                                            logit_values=True) 
                    metrics['val'].append(val_metrics)
                    current_value = val_metrics[metric_for_best_model]
                    lr_scheduler_metric = val_metrics['loss']

                    verbose_string += f'Val: {get_verbose_string(val_metrics)} | '
                else:
                    current_value = train_metrics[metric_for_best_model]
                    lr_scheduler_metric = train_metrics['loss']
                for lr_scheduler in lr_schedulers:
                    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        lr_scheduler.step(lr_scheduler_metric)
                    else:
                        lr_scheduler.step()

                # Evaluate on test set
                if dataloader_test is not None:
                    test_metrics = run_epoch(model=model,
                                             dataloader=dataloader_test,
                                             criterion=criterion,
                                             is_classification=is_classification,
                                             train=False,
                                             metrics=metrics_run_epoch,
                                             prefix=prefix + 'Test',
                                             device=device,) 
                    metrics['test'].append(test_metrics)
                    verbose_string += f'Test: {get_verbose_string(test_metrics)}'

                tqdm.write(verbose_string)

                if early_stopping_coefficient * current_value > early_stopping_coefficient * best_value:
                    best_value = current_value
                    best_model_state_dict = model.state_dict()
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += 1
                    if steps_without_improvement >= patience:
                        break
            
            model.load_state_dict(best_model_state_dict)

            last_training_loss = metrics['train'][-1]['loss']
            if last_training_loss > 0.001 and retrain:
                print('\x1b[1;30;41m' + 'Retraining network due to high loss...' + '\x1b[0m')
                model.load_state_dict(state_copy)

                learning_rate = learning_rate * 1.5

            if last_training_loss < 0.001 or not retrain:
                training_done = True

                if dataloader_val is not None:
                    # get mean and std from validation logits
                    metrics_logits = metrics['val'][-2:] # Get last two epochs
                    metrics_logits = [m['logits'] for m in metrics_logits]
                    # metrics_logits = metrics['val'][-1]['logits']
                    # metrics_logits = np.array(metrics_logits)

                    # ood method: MaxLogit
                    mean_maxlogits = np.mean(metrics_logits)
                    std_maxlogits = np.std(metrics_logits)

                    if std_maxlogits <= 0.5:
                        std_maxlogits = np.abs(mean_maxlogits*0.2)

                    model.networks_maxlogits.append([mean_maxlogits,std_maxlogits])
                    model.networks_labels.append(dataloader_train.dataset.classes)

                if name is not None:
                    torch.save(model.full_state_dict(), os.path.join(model_path, f'full_state_dict_{task}.pt'))

    return metrics

def sigmoid(x):
    return 1 / (1 + np.exp(-x))