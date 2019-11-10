import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def basic_scatter(data, y_label="choice", x_label="time"):

    keys = sorted(data.keys())
    n_keys = len(keys)

    fig, axes = plt.subplots(ncols=n_keys)

    colors = [f'C{i}' for i in range(n_keys)]

    for i in range(n_keys):

        k = keys[i]
        y = data[k]

        ax = axes[i]
        ax.scatter(range(len(y)), y, color=colors[i],
                   alpha=0.2, label=k)

        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.show()

