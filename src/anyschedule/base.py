class BaseScheduler:
    """
    HyperParameter scheduler base class
    start: int, start step, include
    end: int, end step, exclude
    warmup: int, warmup steps
    value: float, initial value

    _get_value: return the value of hyperparameter at "internal" step
    __call__: return the value of hyperparameter at "global" step
    """

    def __init__(
        self,
        start=None,
        end=None,
        warmup=0,
        value=1.0,
        min_value=None,
        init_value=None,
        step_size=1,
        **kwargs,
    ):
        assert value is not None, "value must be provided"
        self.value = value
        self.init_value = init_value
        self.min_value = min_value or 0
        self.start = start or 0
        self.warmup = warmup or 0
        self.end = end or float("inf")
        self.period = (self.end - self.start - self.warmup)//step_size
        self.step_size = step_size

        if self.end == float("inf"):
            self.period = float("inf")

        assert (
            self.period > 0
        ), f"Invalid settings: start={start}, end={end}, warmup={warmup}"
        self._sub_init(**kwargs)

    def _sub_init(self):
        pass

    def _get_value(self, step):
        raise NotImplementedError

    def __call__(self, step):
        step -= self.start
        if step >= self.end:
            raise ValueError(f"Step {step} is out of range [{self.start}, {self.end})")
        if step < self.warmup:
            interp = step / self.warmup
            init_val = self.init_value or 0
            return self.value * interp + init_val * (1 - interp)
        step -= self.warmup
        step //= self.step_size
        return max(self.min_value, self._get_value(step))


registered_schedulers = {}


def register_scheduler(name):
    def register(scheduler):
        if not issubclass(scheduler, BaseScheduler):
            raise ValueError(
                f"Only subclass of BaseScheduler can be registered, got {scheduler}"
            )
        registered_schedulers[name] = scheduler
        return scheduler

    return register
