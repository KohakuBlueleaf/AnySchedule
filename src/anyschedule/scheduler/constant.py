from ..base import BaseScheduler, register_scheduler


@register_scheduler("constant")
class ConstantScheduler(BaseScheduler):
    def _get_value(self, step):
        return self.value
