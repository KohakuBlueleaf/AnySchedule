import matplotlib as mpl
from matplotlib import pyplot as plt
from anyschedule.utils import get_scheduler


if __name__ == "__main__":
    mpl.rcParams["figure.dpi"] = 120
    scheduler = get_scheduler(
        {
            "mode": "composer",
            "schedules": [
                {
                    "mode": "power",
                    "s0": 1000,
                    "b": -0.5,
                    "end": 0.75,
                },
                {
                    "mode": "cosine",
                    "min_value": 0.001,
                    "end": 1.0,
                },
            ],
            "end": 100000,
            "warmup": 1000,
        }
    )
    results = [scheduler(step) for step in range(120_000)]
    plt.figure(figsize=(16, 9))
    plt.plot(results)
    plt.show()
