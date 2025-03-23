from typing import Any
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import toml

from .utils import get_scheduler


class AnySchedule(lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        last_epoch: int = -1,
        verbose="deprecated",
        config: str | dict[str, Any] = {},
    ):
        if isinstance(config, (str, Path)):
            config = toml.load(config)
        self.config = deepcopy(config)
        self.schedulers = {key: get_scheduler(val) for key, val in self.config.items()}
        self.base_param_groups = {
            key: [group[key] for group in optimizer.param_groups]
            for key in self.schedulers
        }
        super(AnySchedule, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key not in {"optimizer", "schedulers"}
        }

    def get_factors(self):
        return {
            key: scheduler(self.last_epoch)
            for key, scheduler in self.schedulers.items()
        }

    def get_lr(self):
        if "lr" in self.schedulers:
            factor = self.schedulers["lr"](self.last_epoch)
            return [val * factor for val in self.base_param_groups["lr"]]
        else:
            return self.base_param_groups["lr"]

    def step(self):
        super(AnySchedule, self).step()
        for key, scheduler in self.schedulers.items():
            if key == "lr":
                continue
            factor = scheduler(self.last_epoch)
            for group, val in zip(
                self.optimizer.param_groups, self.base_param_groups[key]
            ):
                if key not in group:
                    continue
                if isinstance(group[key], torch.Tensor):
                    group[key].fill(factor * val)
                else:
                    group[key] = factor * val
