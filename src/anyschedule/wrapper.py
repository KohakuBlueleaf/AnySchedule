from typing import Any
from pathlib import Path

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
        super(AnySchedule, self).__init__(optimizer, last_epoch)
        if isinstance(config, (str, Path)):
            config = toml.load(config)
        self.config = config
        self.scheduler = get_scheduler(config)

    def get_lr(self):
        factor = self.scheduler(self.last_epoch)
        return [group["lr"] * factor for group in self.optimizer.param_groups]