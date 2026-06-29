import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from configs import Configuration


class BaseLearner(object):
    def __init__(self, configs: Configuration):
        self._cur_domain_id = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self.topk = 5

        self._device = configs.device
        self.args = configs

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T, y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true), decimals=2
        )

        return ret
    
    def eval_task(self):
        DIL_accuracy, DIL_accuracy_with_oracle, domain_classification_accuracy = self._eval_cnn(self.test_loader)

        return DIL_accuracy, DIL_accuracy_with_oracle, domain_classification_accuracy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
