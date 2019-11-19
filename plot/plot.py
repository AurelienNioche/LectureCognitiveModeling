import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plot.part import plot_mean_std, scatter_binary, \
    plot_bar_best_metric, plot_model_metric
from plot.utils import custom_ax
from stats.stats import rolling_mean


def learning_rate(
        model,
        n_iteration=100,
        param_values=(0.01, 0.1, 0.2, 0.3)):

    n_param_values = len(param_values)

    values = np.zeros((n_iteration, n_param_values))

    for i in range(n_param_values):
        alpha = param_values[i]
        agent = model(
            q_learning_rate=alpha,
            q_temp=None)
        for t in range(n_iteration):

            values[t, i] = agent.q_values[0]
            agent.learn(option=0, success=1)

    fig, ax = plt.subplots(figsize=(4, 4))
    lines = ax.plot(values)

    ax.set_xlabel("time")
    ax.set_ylabel("value")

    ax.set_title("Effect of learning rate")

    ax.legend(lines, [r"$\alpha=" + f'{v}$' for v in param_values])

    plt.plot()


def softmax_temperature(param_values=(0.05, 0.25, 0.5, 0.75)):

    n_x_values = 100
    x_values = np.linspace(-1, 1, n_x_values)

    n_param_values = len(param_values)

    values = np.zeros((len(x_values), n_param_values))

    for i in range(n_param_values):
        for j in range(n_x_values):
            x = x_values[j]
            tau = param_values[i]
            values[j, i] = 1 / (1 + np.exp(-x/tau))

    fig, ax = plt.subplots(figsize=(4, 4))
    lines = ax.plot(x_values, values)

    ax.set_xlabel("Q(A) - Q(B)")
    ax.set_ylabel("p(A)")
    ax.set_title("Effect of temperature")

    ax.legend(lines, [r"$\tau=" + f'{v}$' for v in param_values])

    plt.plot()


def behavior_basic(choices, successes, n_option=2):

    # Create figure and axes
    n_rows = 2
    fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2*n_rows))

    # Plot scatter choices
    scatter_binary(ax=axes[0],
                   y=choices,
                   title='Choices',
                   n_option=n_option)

    # PLot scatter successes
    scatter_binary(ax=axes[1],
                   y=np.asarray(successes, dtype=int),
                   y_label="success",
                   scatter_labels=('failure', 'success'),
                   colors=np.array(['red', 'green']),
                   title='Successes',
                   n_option=n_option)

    plt.tight_layout()
    plt.show()


def behavior_average(choices, successes,
                     n_option=2,
                     axes=None):

    n_rows = 4
    if axes is None:
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))
        show = True
    else:
        assert len(axes) == n_rows, \
            f'{len(axes)} axes provided but {n_rows} are required.'
        show = False

    # PLot scatter choices
    ax = axes[0]
    scatter_binary(ax=ax, y=choices, title="Choices", n_option=n_option)
    ax.legend()

    # Plot rolling mean choices
    ax = axes[1]
    for i in range(n_option):
        y = choices == i
        ax.plot(rolling_mean(np.asarray(y, dtype=int)), label=f'option {i}')
    custom_ax(ax=ax, y_label="Freq. choice", title="Choices (freq.)",
              n_iteration=len(choices))

    # Plot scatter successes
    ax = axes[2]
    scatter_binary(ax=ax, y=np.asarray(successes, dtype=int),
                   y_label="success",
                   scatter_labels=('failure', 'success'),
                   colors=np.array(['red', 'green']),
                   title="Successes",
                   n_option=n_option)

    # Plot rolling mean successes
    ax = axes[3]
    y = successes
    ax.plot(rolling_mean(np.asarray(y, dtype=int)))
    custom_ax(ax=ax, y_label="success (freq.)",
              title="Successes (freq.)",
              legend=False,
              n_iteration=len(successes))

    if show:
        plt.tight_layout()
        plt.show()


def latent_variables_rw(q_values, p_choices, choices, successes,
                        axes=None):

    if axes is None:
        n_rows = 6
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))
        show = True
    else:
        show = False

    # Extract from data...
    n_option = q_values.shape[1]

    # Plot q-values
    ax = axes[0]
    lines = ax.plot(q_values)
    ax.legend(lines, [f"option {i}" for i in range(n_option)])
    custom_ax(ax=ax, y_label="value", title="Q-values", legend=False,
              n_iteration=len(q_values))

    # Plot choice probabilities
    ax = axes[1]
    lines = ax.plot(p_choices)
    ax.legend(lines, [f"option {i}" for i in range(n_option)])
    custom_ax(ax=ax, y_label="value",
              title="Probabilities", legend=False,
              n_iteration=len(p_choices))

    # Plot average behavior
    behavior_average(choices=choices, successes=successes,
                     axes=axes[2:])

    if show:
        plt.tight_layout()
        plt.show()


def pop_latent_variables_rw(
        q_values, p_choices, choices, successes, axes=None):

    if axes is None:
        n_rows = 4
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5 * n_rows))
        show = True
    else:
        show = False

    # Extract from data
    n_option = q_values.shape[-1]

    # Plot q-values
    ax = axes[0]
    for i in range(n_option):
        label = f"option {i}"
        y = q_values[:, :, i]
        plot_mean_std(ax=ax, y=y, label=label)

    custom_ax(ax=ax, y_label="value", title="Q-values",
              n_iteration=len(q_values))

    # Plot choice probabilities
    ax = axes[1]
    for i in range(n_option):
        label = f"option {i}"
        y = p_choices[:, :, i]
        plot_mean_std(ax=ax, y=y, label=label)
    custom_ax(ax=ax, y_label='p', title='Probabilities',
              n_iteration=len(p_choices))

    # Plot choice average
    ax = axes[2]
    for i in range(n_option):
        y = choices == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')
    custom_ax(ax=ax, y_label='freq. choice', title="Choices",
              n_iteration=len(choices))

    # Plot success average
    ax = axes[3]
    plot_mean_std(ax=ax, y=np.asarray(successes, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes",
              legend=False,
              n_iteration=len(successes))

    if show:
        plt.tight_layout()
        plt.show()


def pop_comparison_best_fit(data_init, data_best):

    n_cols = 2
    n_rows = 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(4*n_cols,  2.5*n_rows))

    titles = {
        "Initial": axes[0, 0],
        "Best-fit": axes[0, 1],
    }
    for title, ax in titles.items():
        ax.text(0.5, 1.2, title,
                horizontalalignment='center',
                transform=ax.transAxes,
                size=15, weight='bold')

    # Here for comparison
    q_values, p_choices, choices, successes = data_init
    pop_latent_variables_rw(
        q_values=q_values,
        p_choices=p_choices,
        choices=choices,
        successes=successes,
        axes=axes[:, 0])

    # Data with best fit parameters
    q_values, p_choices, choices, successes = data_best
    pop_latent_variables_rw(
        q_values=q_values,
        p_choices=p_choices,
        choices=choices,
        successes=successes,
        axes=axes[:, 1])

    plt.tight_layout()
    plt.show()


def parameter_space_exploration(
        data,
        labels,
        n_levels=100,
        title=None):

    x, y, z = data
    x_label, y_label = labels

    fig, ax = plt.subplots(figsize=(5, 5))

    # Axes labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Title
    ax.set_title(title)

    # Get coordinates
    x_coordinates, y_coordinates = np.meshgrid(x, y)

    # Draw phase diagram
    c = ax.contourf(x_coordinates, y_coordinates, z,
                    levels=n_levels, cmap='viridis')

    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(c, cax=cax)
    cbar.ax.set_ylabel('Log likelihood')

    # Square aspect
    ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


def parameter_recovery(data,
                       x_label='Used to simulate',
                       y_label='Recovered'):

    # Extract data
    keys = sorted(data.keys())
    n_keys = len(keys)

    # Define colors
    colors = [f'C{i}' for i in range(n_keys)]

    # Create fig and axes
    fig, axes = plt.subplots(ncols=n_keys, figsize=(3*n_keys, 3))

    for i in range(n_keys):

        # Select ax
        ax = axes[i]

        # Extract data
        k = keys[i]
        title = k
        x, y = data[k]

        # Create scatter
        ax.scatter(x, y, alpha=0.5, color=colors[i])

        # Set axis label
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Set title
        ax.set_title(title)

        # Set ticks positions
        ax.set_xticks((0, 0.5, 1))
        ax.set_yticks((0, 0.5, 1))

        # Plot identity function
        ax.plot(range(2), linestyle="--", alpha=0.2, color="black", zorder=-10)

        # Square aspect
        ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


def confusion_matrix(data, tick_labels,
                     title="Confusion matrix",
                     x_label="Used to simulate",
                     y_label="Recovered model"):
    norm_data = np.zeros(shape=data.shape)

    # Normalize
    for i in range(len(data)):
        norm_data[i] = data[i] / np.sum(data[i])

    # Define ticks labels
    x_tick_labels = tick_labels
    y_tick_labels = tick_labels

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(2*data.shape[1],
                                    2*data.shape[0]))

    # Draw matrix
    im = ax.imshow(norm_data, alpha=0.5)

    # Set axis labels
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_tick_labels)))
    ax.set_yticks(np.arange(len(y_tick_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_tick_labels)
    ax.set_yticklabels(y_tick_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # ... and move them again
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_tick_labels)):
        for j in range(len(x_tick_labels)):
            value = f'{norm_data[i, j]:.3f}'
            ax.text(j, i, value,
                    ha="center", va="center", color="black")

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    # Set title
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


def fake_xp(choices, successes,
            choices_bf, successes_bf,
            lls,
            bic_scores,
            lls_freq,
            lls_err,
            bic_freq,
            bic_err,
            n_option,
            model_names):

    n_rows = 8
    fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5 * n_rows))

    # Plot average
    ax = axes[0]
    for i in range(n_option):
        y = choices == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')
    custom_ax(ax=ax, y_label='freq. choice', title="Choices",
              n_iteration=len(choices))

    # Plot successes
    ax = axes[1]
    plot_mean_std(ax=ax, y=np.asarray(successes, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes",
              legend=False,
              n_iteration=len(successes))

    # Plot LLS - scatterplot + boxplot
    ax = axes[2]
    plot_model_metric(ax=ax, metrics=lls,
                      title="Log-Likelihood Sums",
                      y_label="LLS",
                      x_tick_labels=model_names)

    # Plot LLS - barplot
    ax = axes[3]
    plot_bar_best_metric(ax=ax, freq=lls_freq, y_err=lls_err,
                         y_label='Highest LLS (freq.)',
                         title="Highest LLS",
                         x_tick_labels=model_names)

    # Plot BIC - scatterplot + boxplot
    ax = axes[4]
    plot_model_metric(ax=ax, metrics=bic_scores,
                      title="BIC Scores",
                      y_label="BIC",
                      x_tick_labels=model_names)
    ax.invert_yaxis()

    # Plot BIC - barplot
    ax = axes[5]
    plot_bar_best_metric(ax=ax, freq=bic_freq, y_err=bic_err,
                         y_label='Lowest BIC (freq.)',
                         title="Lowest BIC",
                         x_tick_labels=model_names)

    # Plot best-fit choice average
    ax = axes[6]
    for i in range(n_option):
        y = choices_bf == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')
    custom_ax(ax=ax, y_label='freq. choice', title="Choices Best-fit",
              n_iteration=len(choices_bf))

    # Plot best-fit success average
    ax = axes[7]
    plot_mean_std(ax=ax, y=np.asarray(successes_bf, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes Best-fit",
              legend=False,
              n_iteration=len(successes_bf))

    plt.tight_layout()
    plt.show()


def comparison_best_fit_rw(
        q_values,
        p_choices,
        choices,
        successes,
        q_values_bf,
        p_choices_bf,
        choices_bf,
        successes_bf
):

    n_cols = 2
    n_rows = 6
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(4*n_cols,  2.5*n_rows))

    titles = {
        "Initial": axes[0, 0],
        "Best-fit": axes[0, 1]
    }
    for title, ax in titles.items():
        ax.text(0.5, 1.2, title,
                horizontalalignment='center',
                transform=ax.transAxes,
                size=15, weight='bold')

    # Here for comparison
    latent_variables_rw(
        q_values=q_values,
        p_choices=p_choices,
        choices=choices,
        successes=successes,
        axes=axes[:, 0])

    # New simulation with best fit parameters
    latent_variables_rw(
        q_values=q_values_bf,
        p_choices=p_choices_bf,
        choices=choices_bf,
        successes=successes_bf,
        axes=axes[:, 1])

    plt.tight_layout()
    plt.show()
