from .base import registered_schedulers


def get_scheduler(config: dict):
    scheduler = config.pop("mode")
    return registered_schedulers[scheduler](**config)
