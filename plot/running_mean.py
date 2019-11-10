import matplotlib.pyplot as plt
import pandas as pd


def running_mean(data, y_label='choice', x_label="time", window=20):

    keys = sorted(data.keys())
    n_keys = len(keys)

    fig, axes = plt.subplots(ncols=n_keys)

    colors = [f'C{i}' for i in range(n_keys)]

    for i in range(n_keys):

        k = keys[i]

        y = data[k]

        ax = axes[i]
        ax.plot(pd.Series(y).rolling(window).mean(),
                color=colors[i], alpha=0.2, label=k)
        ax.set_ylim(-0.02, 1.02)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.legend()
    plt.tight_layout()
    plt.show()
