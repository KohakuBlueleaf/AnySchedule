import toml
from matplotlib import pyplot as plt

from anyschedule.utils import get_scheduler


if __name__ == "__main__":
    scheduler = get_scheduler(toml.load("config/example.toml")["lr"])
    results = [scheduler(step) for step in range(2000)]
    plt.subplot(2, 1, 1)
    plt.plot(results)
    scheduler = get_scheduler(toml.load("config/example-wsd.toml")["lr"])
    results = [scheduler(step) for step in range(2000)]
    plt.subplot(2, 1, 2)
    plt.plot(results)
    plt.show()