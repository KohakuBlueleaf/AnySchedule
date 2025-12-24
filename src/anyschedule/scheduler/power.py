from ..base import BaseScheduler, register_scheduler


@register_scheduler("power")
class PowerScheduler(BaseScheduler):
    def _sub_init(self, s0=10, b=-0.5):
        assert s0 > 0, "s0 must be greater than 0"
        assert b < 0, "b must be less than 0"
        self.s0 = float(s0)
        self.b = float(b)

    def _get_value(self, step):
        mult = ((step + self.s0) / self.s0) ** self.b
        return max(self.min_value, self.value * mult)


if __name__ == "__main__":
    scheduler = PowerScheduler(start=0, end=None, warmup=10, value=0.1, s0=10, b=-0.5)
    for step in range(100):
        print(f"step {step}: {scheduler(step)}")
