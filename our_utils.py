import torch
import random
import torch.nn as nn
import numpy as np
import time
from torch import Tensor as T
from torch.nn import functional as F
import collections.abc as abc
from torch.utils.data import Dataset
from torch import optim
import subprocess
import shutil
import os
import json
import math
from torch.distributions import MultivariateNormal


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x


def get_time_str(add_time: bool = True):
    my_str = 'Date_%Y-%m-%d'
    if add_time:
        my_str += ',Time_%H-%M-%S'
    return time.strftime(my_str, time.localtime())


def set_seed(seed):
    """Sets the seed of random number generators to the predefined seed number for reproducibility.
    """
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


def freeze_or_unfreeze(obj: nn.Module, requires_grad: bool):
    if obj is None:
        print("WARNING: The object is None. It has no parameter to freeze!")
        return
    if isinstance(obj, nn.Parameter):
        obj.requires_grad = requires_grad
    
    for param in obj.parameters():
        param.requires_grad = requires_grad
    
    
def are_consecutive(numbers_list: list):
    if len(numbers_list) < 2:
        return True
    
    sorted_numbers = sorted(numbers_list)
    
    for i in range(len(sorted_numbers) - 1):
        if sorted_numbers[i + 1] != sorted_numbers[i] + 1:
            return False
    
    return True


def to_device(
    input: T | abc.Mapping | abc.Sequence | set | int | str | float,
    device
):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str) or isinstance(input, int) or isinstance(input, float) or input is None:
        return input
    elif isinstance(input, abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    elif isinstance(input, set):
        return {to_device(itm, device=device) for itm in input}
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")
    
    
def get_object_name(obj):
    for name, value in globals().items():
        if value is obj:
            return name
    
    
class AverageAccuracyCalculator:
    def __init__(self):
        self.count = 0
        self.sum = 0.0

    def update(self, labels_inferenced: T, labels_real: T):
        self.sum += ((labels_inferenced == labels_real).float()).sum().item()
        self.count += len(labels_real) * 1.0

    def calculate(self):
        if self.count > 0:
            return 100.0 * self.sum / self.count
        else:
            return -1.0


def get_params_groups(
    model: nn.Module | nn.Parameter,
    name_model: str,
    lr:float = 0.0,
    weight_decay: float=None,
    disable_weight_decay=False
):
    if model is None or lr == 0:
        return []
    regularized = []
    not_regularized = []
    
    if isinstance(model, nn.Parameter):
        if model.requires_grad:
            not_regularized.append(model)
    else:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1 or param.numel() == 1 or disable_weight_decay:
                not_regularized.append(param)
            else:
                regularized.append(param)
    
    regularized_dict = {'params': regularized, 'name': name_model + ' regularized'}
    not_regularized_dict = {'params': not_regularized, 'weight_decay': 0., 'name': name_model + ' not_regularized'}
    
    if lr != -1:
        regularized_dict['lr'] = lr
        not_regularized_dict['lr'] = lr
        
    if weight_decay is not None:
        regularized_dict['weight_decay'] = weight_decay
        
    result = []
    
    if len(regularized_dict['params']) > 0:
        result.append(regularized_dict)
    
    if len(not_regularized_dict['params']) > 0:
        result.append(not_regularized_dict)
    
    return result


def show_number_of_parameters_in_pramas_groups(params_all: list, logger):
    num_parameters = 0
    
    for param_list1 in params_all:
        num_parameters_comp = 0
        
        for p in param_list1['params']:
            if p.requires_grad:
                num_parameters_comp += p.numel()
    
        num_parameters += num_parameters_comp
        logger.info(f"Number of learnable parameters of {param_list1['name']}: {num_parameters_comp}")
    
    logger.info(f'Total number of learnable parameters: {num_parameters}')
    
    
def get_optimizer_from_params(params_all, optimizer_name: str, lr_default: float, weight_decay: float):
    assert len(params_all) > 0
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(
            params_all,
            momentum=0.9,
            lr=lr_default,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(
            params_all,
            lr=lr_default,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(
            params_all,
            lr=lr_default,
            weight_decay=weight_decay
        )
    else:
        raise NotImplementedError()
        
    return optimizer

def calculate_mean_and_std_for_a_list(numbers_list: list):
    mean = np.mean(numbers_list)
    std = np.std(numbers_list)
    
    return mean, std


def get_printable_string_from_a_list_of_float_numbers_with_two_digits(numbers_list: list):
    results = [f'{acc:.2f}' for acc in numbers_list]
        
    results = ', '.join(results)
    
    results = '[' + results + ']'
    
    mean, std = calculate_mean_and_std_for_a_list(numbers_list)
    
    return results, mean, std


def highlighted_message(message: str, max_length: int = 60):
    assert len(message) < max_length - 4
    margin = max_length - 4 - len(message) // 2   # :)
    result = '-' * margin + ' ' + message + ' ' + '-' * margin
    return result


def print_matrix(text_features: T, num_columns: int = 10):
    assert text_features.dim() == 2
    
    # print('[')
    for i in range(text_features.shape[0]):
        print('[', end='')
        for j in range(num_columns):
            print(f"{text_features[i, j].item():3f}", end='')
            if j < num_columns - 1:
                print(', ', end='')
        print(']')
    # print(']')
    
    
def similarity(features_1: T, features_2: T, cosine: bool = True):
    # cosine: We normalize the embeddings. Therefore, we ignore the magnitudes and only consider the angles.
    assert 1 <= features_1.dim() <= 2 and 1 <= features_2.dim() <= 2
    
    if features_1.dim() == 1:
        features_1 = features_1.unsqueeze(0)
        
    if features_2.dim() == 1:
        features_2 = features_2.unsqueeze(0)
    
    if cosine:       # We consider the angle only
        features_1_norm = F.normalize(features_1, dim=-1)
        features_2_norm = F.normalize(features_2, dim=-1)
        logits_features_1_per_feature_2 = features_1_norm @ features_2_norm.t()
    else:
        logits_features_1_per_feature_2 = features_1 @ features_2.t()
    
    # logits_features_2_per_feature_1 = logits_features_1_per_feature_2.t()
    return logits_features_1_per_feature_2      # , logits_features_2_per_feature_1


def obtain_driver_version():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
        capture_output=True, text=True
    )
    driver_version = result.stdout.strip()
    
    return driver_version


def print_overwrite(text):
    print(" " * shutil.get_terminal_size().columns, end='\r')
    print(text, end='\r')


def power_norm(
    data: T,
    alpha: nn.Parameter | float | T = 0.5
):
    """Applies sign-preserved power normalization."""
    # Add a tiny epsilon to prevent NaNs when taking the derivative of 0 (if gradients were enabled)
    return torch.sign(data) * torch.pow(torch.abs(data) + 1e-8, alpha)


def obtain_class_names(dataset_name: str):
    class_names_real_json_file_path = 'descriptions/class_names_real.json'
    class_names_json_file_path = 'descriptions/class_names.json'
    
    if os.path.exists(class_names_real_json_file_path):
        with open(class_names_real_json_file_path, 'r') as file:
            data = json.load(file)
            class_names_real = np.array(data[dataset_name])
    else:
        raise Exception("Error: The class names file is required!")
    
    if os.path.exists(class_names_json_file_path):
        with open(class_names_json_file_path, 'r') as file:
            data = json.load(file)
            class_names = np.array(data[dataset_name])
    else:
        raise Exception("Error: The class names file is required!")
    
    return class_names_real, class_names


def interpolate(
    coef: float | T | nn.Parameter,
    a: T,
    b: T,
    normalize: bool = True
) -> T:
    """It computes res = coef * a + (1.0 - coef) * b

    Args:
        a (T): _description_
        b (T): _description_
        coef (float): _description_

    Returns:
        _type_: _description_
    """
    
    if isinstance(coef, T):
        # assert len(coef) == len(a) == len(b)
        res = coef * a + (1.0 - coef) * b
    elif coef != 0.0 and coef != 1.0:
        res = coef * a + (1.0 - coef) * b
    elif coef == 1.0:
        res = coef * a
    elif coef == 0.0:
        res = (1.0 - coef) * b
        
    if normalize:
        res = F.normalize(res, dim=-1)
    
    return res


class EstimatedRemainingTime:
    def __init__(self, total_tasks: int):
        """Estimated remaining time

        Args:
            total_tasks (int): Total tasks to be performed
        """
        self.stopwatch = Stopwatch(['total'])
        self.total_tasks = total_tasks
        
    def calculate(self, num_finished_tasks: int) -> str:
        
        result = self.stopwatch.calculate_estimate_remaining_time(key='total', total_tasks=self.total_tasks, num_finished_tasks=num_finished_tasks)
        
        return result
    
    def reset(self):
        self.stopwatch.reset('total')
        
        
class Stopwatch:
    """
    Stopwatch computes the time between start and stop.
    Then we can add time to the total_elapsed_time dictionary by watch name.
    """
    def __init__(self, keys: list = None):
        if keys is None:
            keys = []
        self._start_time = {k: time.time() for k in keys}

    def reset(self, key):
        self._start_time[key] = time.time()

    def elapsed_time(self, key):
        if key in self._start_time:
            return time.time() - self._start_time[key]

        self.reset(key)
        return 0.0

    @staticmethod
    def convert_to_hours_minutes(time_in_seconds: float) -> str:
        time_in_seconds = int(time_in_seconds)
        days = time_in_seconds // (24 * 3600)
        hours = (time_in_seconds % (24 * 3600)) // 3600
        minutes = (time_in_seconds % 3600) // 60
        seconds = time_in_seconds % 60

        def plural(x):
            if x != 1:
                return 's'
            return ''

        res_list = []

        if days > 0:
            res_list.append(f"{days} day{plural(days)}")
        if hours > 0:
            res_list.append(f"{hours} hour{plural(hours)}")
        if minutes > 0:
            res_list.append(f"{minutes} minute{plural(minutes)}")
        res_list.append(f"{seconds} second{plural(seconds)}")

        if len(res_list) == 1:
            return res_list[0]
        elif len(res_list) == 2:
            return ' and '.join(res_list)
        else:
            res = ', '.join(res_list[:-1]) + f', and {res_list[-1]}'
            
        return res

    def elapsed_time_in_hours_minutes(self, key):
        return self.convert_to_hours_minutes(self.elapsed_time(key))
    
    def calculate_estimate_remaining_time(self, key, total_tasks, num_finished_tasks):
        total_time = self.elapsed_time(key)
        remaining_time_str = estimated_remaining_time_string(total_time=total_time, total_tasks=total_tasks, num_finished_tasks=num_finished_tasks)
        return remaining_time_str

    def __getitem__(self, name):
        return self.elapsed_time(name)

    def __getattr__(self, name: str):
        return self.elapsed_time(name)
    
    
def estimated_remaining_time_string(total_time, total_tasks, num_finished_tasks: float):
    num_finished_tasks_from_one = num_finished_tasks + 1.0
    ert = (total_tasks - num_finished_tasks_from_one) * total_time / num_finished_tasks_from_one
    res = "ERT: %s" % Stopwatch.convert_to_hours_minutes(ert)
    return res


def sample_from_gaussian(mean: T, covariance: T, num_samples: int, epsilon: float = 1e-3):
    # mean.shape:       [1, dim_embed]
    # covariance.shape: [dim_embed, dim_embed]
    jitter = epsilon * torch.eye(covariance.shape[0], device=mean.device)     # for the non-positive-definite case
    mvn = MultivariateNormal(mean, covariance_matrix=covariance + jitter)
    
    return mvn.rsample((num_samples,))


@torch.no_grad()
def kl_divergence_fast(base_means, base_covs, cand_means, cand_covs, eps=1e-3):
    """
    base_means:  [L, d]
    base_covs:   [L, d, d]
    cand_means:  [M, d]
    cand_covs:   [M, d, d]
    Returns:     [M] (sum of KL over base classes for each candidate)
    """
    d = base_means.shape[-1]
    identity_scaled = eps * torch.eye(d, device=base_means.device)

    # Precompute inverses and log-dets for candidates
    cand_covs_reg = cand_covs + identity_scaled
    cand_inv = torch.linalg.inv(cand_covs_reg)             # [M, d, d]
    cand_logdet = torch.linalg.slogdet(cand_covs_reg)[1]   # [M]

    base_covs_reg = base_covs + identity_scaled
    base_logdet = torch.linalg.slogdet(base_covs_reg)[1]   # [L]

    trace = torch.einsum('mij,lji->lm', cand_inv, base_covs_reg)

    diff = base_means.unsqueeze(1) - cand_means.unsqueeze(0)  # [L, M, d]
    mahal = torch.einsum('lmd,mde,lme->lm', diff, cand_inv, diff)

    # log-det ratio
    ratio = cand_logdet.unsqueeze(0) - base_logdet.unsqueeze(1)  # [L, M]

    kl = 0.5 * (ratio - d + trace + mahal)  # [L, M]
    return kl.sum(dim=0)  # [M]


class MovingAverageDict:
    def __init__(self, capacity, logger=None):
        self.capacity = capacity
        self.meters: dict[str, _MovingAverage] = {}
        self.logger = logger
        self.logging_method = print if logger is None else logger.info

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key].calculate()
        msg = f"Error: You didn't define or add a value to the {key} key!"
        if self.logger is not None:
            self.logger.exception(msg)
        raise Exception(msg)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = _MovingAverage(self.capacity)
            self.meters[k].update(v)

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def display(self, logger=None):
        for key, val in self.meters.items():
            output = f"Average {key} for the last {val.count} iterations: {val.calculate()}"
            self.logging_method(output)


class _MovingAverage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0.0

    def update(self, x):
        self.sum += x - self.array[self.ind]
        self.array[self.ind] = x
        self.ind = (self.ind + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def calculate(self):
        return self.sum / self.count

    def reset(self):
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0
    

# References:
# https://github.com/lambor9973/cds
# https://github.com/Naeem-Paeedeh/CPLSR