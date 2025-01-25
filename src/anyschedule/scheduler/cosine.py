import math

from ..base import BaseScheduler, register_scheduler


@register_scheduler("cosine")
class CosineScheduler(BaseScheduler):
    def _get_value(self, step):
        if step <= 0:
            return self.value
        if step >= self.period:
            return self.min_value
        return self.min_value + 0.5 * (self.value - self.min_value) * (
            1 + math.cos(math.pi * step / self.period)
        )


if __name__ == "__main__":
    scheduler = CosineScheduler(0, 100, warmup=10, value=0.1, min_value=0.01)
    for step in range(100):
        print(f"step {step}: {scheduler(step)}")
