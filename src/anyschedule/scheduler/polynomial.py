import math

from ..base import BaseScheduler, register_scheduler


@register_scheduler("polynomial")
class PolynomialScheduler(BaseScheduler):
    def _sub_init(self, power=1, closed_form=True):
        self.power = power
        self.closed_form = closed_form

    def _get_value(self, step):
        decay = (
            (1 - (step / self.period))
            / (1 if self.closed_form else (1 - (step - 1) / self.period))
        ) ** self.power
        return (self.value - self.min_value) * decay + self.min_value
