from ..base import BaseScheduler, register_scheduler


@register_scheduler("step")
class StepScheduler(BaseScheduler):
    def _sub_init(
        self, gamma=0.95
    ):
        self.gamma = gamma

    def _get_value(self, step):
        return self.value * (self.gamma**step)
