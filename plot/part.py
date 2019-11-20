import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_mean_std(ax, label=None, y=None):

    # Compute mean and std
    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    # Plot the mean
    ax.plot(mean, label=label)

    # Draw the area mean-STD, mean+STD
    ax.fill_between(
        range(len(mean)),
        mean - std,
        mean + std,
        alpha=0.2
    )


def scatter_binary(ax, y, n_option,
                   colors=None,
                   scatter_labels=None,
                   y_label="choice", title=None):

    # Extract from data...
    n_iteration = len(y)

    # Define colors
    if colors is None:
        colors = np.array([f"C{i}"for i in range(n_option)])

    # Define labels for legend
    if scatter_labels is None:
        scatter_labels = [f'option {i}' for i in range(n_iteration)]

    # Plot the scatter
    ax.scatter(range(len(y)), y+np.random.uniform(-0.2, 0.2, size=len(y)),
               color=colors[y],
               alpha=0.2, s=10)

    # Plot extra points for the legend (trick)
    for i, color in enumerate(colors):
        ax.scatter(-1, -1, color=color, alpha=0.2,
                   label=scatter_labels[i],
                   s=20)

    # Ensure that ticks comprised only integer values
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Place ticks sparsely
    ax.set_xticks((0, int(n_iteration/2), n_iteration))
    ax.set_yticks((0, 1))

    # Set the limits
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.02*n_iteration, n_iteration*1.02)

    # Set the labels
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)

    # Set the figure proportions
    ax.set_aspect(n_iteration*0.1)

    # Set the title
    ax.set_title(title)

    # Put legend
    ax.legend()


def plot_bar_best_metric(ax, freq, y_err,
                         x_tick_labels,
                         y_label, title):

    # Get the ticks
    x_pos = np.arange(len(freq))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_tick_labels)

    # Plot the bar
    ax.bar(x_pos, freq, yerr=y_err)

    # Add label indicating the frequence on top of each bar
    for i in range(len(freq)):

        ax.text(x=i, y=freq[i]+y_err[1, i]+0.05,
                s=f'{freq[i]:.2f}', size=6,
                horizontalalignment='center',
                )

    # Set the y ticks
    ax.set_yticks((0, 0.5, 1))

    # Set the y-limits
    ax.set_ylim((0, max(1, np.max(freq[:]+y_err[1, :])+0.15)))

    # Set the label of the y-axis
    ax.set_ylabel(y_label)

    # Set the title
    ax.set_title(title)


def plot_scatter_metric(data, ax, y_label, x_tick_labels, title):

    # Extract from data...
    n = data.shape[-1]

    # Colors
    colors = np.array([f"C{i}" for i in range(n)])

    # Containers for boxplot
    positions = list(range(n))
    values_box_plot = [[] for _ in range(n)]

    # Containers for scatter
    x_scatter = []
    y_scatter = []
    colors_scatter = []

    # For each boxplot
    for i in range(n):

        # For every value
        for v in data[:, i]:

            # Add value to the boxplot container
            values_box_plot[i].append(v)

            # Add value to the scatter plot
            x_scatter.append(i + np.random.uniform(-0.05*n, 0.05*n))
            y_scatter.append(v)
            colors_scatter.append(colors[i])

    # Plot the scatter
    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=20, alpha=0.2,
               linewidth=0.0, zorder=1)

    # Plot the boxplot
    bp = ax.boxplot(values_box_plot, positions=positions,
                    labels=x_tick_labels, showfliers=False, zorder=2)

    # Set the color of the boxplot
    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')

    # Set the label of the y axis
    ax.set_ylabel(y_label)

    # Set the title
    ax.set_title(title)
