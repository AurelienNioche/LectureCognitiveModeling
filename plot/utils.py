def custom_ax(ax, y_label, n_iteration, title=None, legend=True):

    ax.set_xticks((0, int(n_iteration/2), n_iteration))
    ax.set_yticks((0, 0.5, 1))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02*n_iteration, n_iteration*1.02)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if legend:
        ax.legend()
