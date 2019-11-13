"""
Modeling of decision-making
"""

# =================================================================
# Import your modules =============================================
# =================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import scipy.optimize
import statsmodels.stats
import statsmodels.stats.proportion
import scipy.stats
from tqdm.autonotebook import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.decorator import use_pickle

# =================================================================
# Globals =========================================================
# =================================================================

EPS = np.finfo(np.float).eps

# ======================================================================
# Design your task =====================================================
# ======================================================================

N = 2
P = np.array([0.5, 0.75])
T = 500

# ======================================================================
# Design your model(s) but also competitive models =====================
# ======================================================================


class Random:
    """
    No-learning model
    """
    param_labels = ()
    bounds = ()

    def __init__(self, n_option):
        self.n_option = n_option
        self.options = np.arange(n_option)

    def choose(self):
        p = self.decision_rule()
        return np.random.choice(self.options, p=p)

    def learn(self, option, success):
        self.updating_rule(option=option, success=success)

    def decision_rule(self):
        return np.ones(self.n_option) / self.n_option

    def updating_rule(self, option, success):
        pass


class RL(Random):
    """
    Reinforcement learning model
    """
    param_labels = ("alpha", "beta")
    bounds = (0.01, 1), (0.05, 1)

    def __init__(self, learning_rate, temp, n_option, initial_value=0.5):
        super().__init__(n_option=n_option)
        self.values = np.full(n_option, initial_value)
        self.learning_rate = learning_rate
        self.temp = temp

    def decision_rule(self):

        p_soft = np.exp(self.values / self.temp) / \
             np.sum(np.exp(self.values / self.temp), axis=0)
        return p_soft

    def updating_rule(self, option, success):
        self.values[option] += \
            self.learning_rate * (success - self.values[option])


MODELS = Random, RL

# Study the effect of your parameters ========================================


def plot_learning_parameter(n_iteration=100,
                            param_values=(0.01, 0.1, 0.2, 0.3)):

    n_param_values = len(param_values)

    values = np.zeros((n_iteration, n_param_values))

    for i in range(n_param_values):
        alpha = param_values[i]
        agent = RL(learning_rate=alpha,
                   temp=0.0, n_option=1)
        for t in range(n_iteration):

            values[t, i] = agent.values[0]
            agent.learn(option=0, success=1)

    fig, ax = plt.subplots(figsize=(4, 4))
    lines = ax.plot(values)

    ax.set_xlabel("time")
    ax.set_ylabel("value")

    ax.set_title("Effect of learning rate")

    ax.legend(lines, [r"$\alpha=" + f'{v}$' for v in param_values])

    plt.plot()


def plot_temperature(param_values=(0.05, 0.25, 0.5, 0.75)):

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


plot_learning_parameter()
plot_temperature()

# =================================================================
# First artificial experiment =====================================
# =================================================================

# Produce data ----------------------------------------------------

# Param for the artificial subject
SEED = 0
PARAM_RL = (0.1, 0.1)


# class UsePickle(object):
#
#     def __init__(self, f):
#         print("inside my_decorator.__init__()")
#         self.f = f
#
#     def __call__(self, *args, **kwargs):
#         self.f(*args, **kwargs)
#         print("inside my_decorator.__call__()")


@use_pickle
def run_simulation(agent_model, param=(), seed=None, force=False):

    np.random.seed(seed)

    agent = agent_model(n_option=N, *param)

    choices = np.zeros(T, dtype=int)
    successes = np.zeros(T, dtype=bool)

    # Simulate the task
    for t in range(T):
        # Determine choice
        choice = agent.choose()

        # Determine success
        p_success = P[choice]
        success = np.random.choice(
            [0, 1],
            p=np.array([1 - p_success, p_success]))

        # Make agent learn
        agent.learn(option=choice, success=success)

        # Backup
        choices[t] = choice
        successes[t] = success

    return choices, successes


CHOICES, SUCCESSES = \
    run_simulation(agent_model=RL, param=PARAM_RL, seed=SEED)


# Plot ---------------------------------------------------------------------

def custom_ax(ax, y_label, title=None, legend=True):

    ax.set_xticks((0, int(T/2), T))
    ax.set_yticks((0, 0.5, 1))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.02*T, T*1.02)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)

    if title is not None:
        ax.set_title(title)

    if legend:
        ax.legend()


def plot_mean_std(ax, label=None, y=None):

    mean = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    ax.plot(mean, label=label)
    ax.fill_between(
        range(T),
        mean - std,
        mean + std,
        alpha=0.2
    )


def scatter_choices(ax, y,
                    colors=None,
                    scatter_labels=None,
                    y_label="choice", title=None):

    if colors is None:
        colors = np.array([f"C{i}"for i in range(N)])

    if scatter_labels is None:
        scatter_labels = [f'option {i}' for i in range(N)]

    ax.scatter(range(len(y)), y, color=colors[y],
               alpha=0.2, s=10)

    for i, color in enumerate(colors):
        ax.scatter(-1, -1, color=color, alpha=0.2,
                   label=scatter_labels[i],
                   s=20)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    custom_ax(ax, y_label=y_label, title=title)


def rolling_mean(y, window=50):
    return pd.Series(y).rolling(window).mean()


def plot_behavior_basic(choices, successes):

    n_rows = 2
    fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))

    scatter_choices(ax=axes[0],
                    y=choices,
                    title='Choices')

    scatter_choices(ax=axes[1],
                    y=np.asarray(successes, dtype=int),
                    y_label="success",
                    scatter_labels=('failure', 'success'),
                    colors=np.array(['red', 'green']),
                    title='Successes')

    plt.tight_layout()
    plt.show()


# Begin by the more basic possible
plot_behavior_basic(choices=CHOICES, successes=SUCCESSES)


def plot_behavior_average(choices, successes, axes=None):

    n_rows = 4
    if axes is None:
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))
        show = True
    else:
        assert len(axes) == n_rows, \
            f'{len(axes)} axes provided but {n_rows} are required.'
        show = False

    ax = axes[0]
    scatter_choices(ax=ax, y=choices, title="Choices")
    ax.legend()

    ax = axes[1]
    for i in range(N):
        y = choices == i
        ax.plot(rolling_mean(np.asarray(y, dtype=int)), label=f'option {i}')
    custom_ax(ax=ax, y_label="Freq. choice", title="Choices (freq.)")

    ax = axes[2]
    scatter_choices(ax=ax, y=np.asarray(successes, dtype=int),
                    y_label="success",
                    scatter_labels=('failure', 'success'),
                    colors=np.array(['red', 'green']),
                    title="Successes")

    ax = axes[3]
    y = successes
    ax.plot(rolling_mean(np.asarray(y, dtype=int)))
    custom_ax(ax=ax, y_label="success (freq.)",
              title="Successes (freq.)",
              legend=False)

    if show:
        plt.tight_layout()
        plt.show()


# # ...then maybe you can do better
plot_behavior_average(choices=CHOICES, successes=SUCCESSES)


# # (Behavioral) Stats =======================================================
#
def format_p(p, threshold=0.05):

    pf = f'={p:.3f}' if p >= 0.001 else '<0.001'
    pf += " *" if p <= threshold else " NS"
    return pf

#
# def stats():
#
#     n_models = len(MODELS)
#
#     contingency_table = np.zeros((n_models, N))
#
#     for i in range(n_models):
#
#         m = MODELS[i]
#         m_name = m.__name__
#         choices = HIST_CHOICES[m_name]
#
#         k, n = np.sum(choices), len(choices)
#
#         obs = n-k, k
#
#         ci_low, ci_upp = \
#             statsmodels.stats.proportion.proportion_confint(count=k, nobs=n)
#
#         print(f"Model: {m_name}")
#         print(f"prop choose best= {k/n:.3f}, CI=[{ci_low:.3f}, {ci_upp:.3f}]")
#
#         chi2, p = scipy.stats.chisquare(obs)
#         print("Chi2 for equality of proportion")
#         print(f"Chi2={chi2:.3f}, p{format_p(p)}")
#         print()
#
#         contingency_table[i] = obs
#
#     chi2, p, dof, ex = scipy.stats.chi2_contingency(contingency_table,
#                                                     correction=False)
#     print("Chi2 for independence")
#     print(f"Chi2={chi2:.3f}, p{format_p(p)}")
#     print()
#
#
# stats()


# =======================================================================
# Latent variable =======================================================
# =======================================================================

def data_latent_variables(choices, successes, model, param):

    agent = model(n_option=N, *param)

    assert hasattr(agent, "values"), \
        "Model instance needs to have 'values' attribute"

    q_values = np.zeros((T, N))
    p_choices = np.zeros((T, N))

    # (Re-)Simulate the task
    for t in range(T):

        # Register values
        q_values[t] = agent.values

        # Register probablility of choices
        p_choices[t] = agent.decision_rule()

        # Make agent learn
        agent.learn(option=choices[t],
                    success=successes[t])

    return q_values, p_choices


def plot_latent_variables(q_values, p_choices, choices, successes,
                          axes=None):

    if axes is None:
        n_rows = 6
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))
        show = True
    else:
        show = False

    # Plot values
    ax = axes[0]

    lines = ax.plot(q_values)
    ax.legend(lines, [f"option {i}" for i in range(N)])

    custom_ax(ax=ax, y_label="value", title="Q-values", legend=False)

    # Plot probablilities
    ax = axes[1]

    lines = ax.plot(p_choices)

    ax.legend(lines, [f"option {i}" for i in range(N)])

    custom_ax(ax=ax, y_label="value",
              title="Probabilities", legend=False)

    plot_behavior_average(choices=choices, successes=successes,
                          axes=axes[2:])

    if show:
        plt.tight_layout()
        plt.show()


Q_VALUES, P_CHOICES = data_latent_variables(
        choices=CHOICES,
        successes=SUCCESSES,
        model=RL,
        param=PARAM_RL
    )

plot_latent_variables(q_values=Q_VALUES,
                      p_choices=P_CHOICES,
                      successes=SUCCESSES,
                      choices=CHOICES)


# ========================================================================
# Population simulation
# ========================================================================


@use_pickle
def population_simulation(model, param, n=30):

    pop_q_values = np.zeros((n, T, N))
    pop_p_choices = np.zeros((n, T, N))
    pop_choices = np.zeros((n, T), dtype=int)
    pop_successes = np.zeros((n, T), dtype=bool)

    for i in range(n):

        choices, successes \
            = run_simulation(agent_model=model, param=param, force=True)

        q_values, p_choices \
            = data_latent_variables(
                choices=choices, successes=successes,
                model=model,
                param=param)

        pop_q_values[i] = q_values
        pop_p_choices[i] = p_choices
        pop_choices[i] = choices
        pop_successes[i] = successes

    return pop_q_values, pop_p_choices, pop_choices, pop_successes


def plot_pop_latent_variables(
        q_values, p_choices, choices, successes, axes=None):

    if axes is None:
        n_rows = 4
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5 * n_rows))
        show = True
    else:
        show = False

    # Plot values
    ax = axes[0]

    assert q_values.shape[-1] == N

    for i in range(N):

        label = f"option {i}"
        y = q_values[:, :, i]

        plot_mean_std(ax=ax, y=y, label=label)

    custom_ax(ax=ax, y_label="value", title="Q-values")

    # Plot probablilities
    ax = axes[1]

    for i in range(N):

        label = f"option {i}"
        y = p_choices[:, :, i]

        plot_mean_std(ax=ax, y=y, label=label)

    custom_ax(ax=ax, y_label='p', title='Probabilities')

    # Plot average
    ax = axes[2]

    for i in range(N):
        y = choices == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')

    custom_ax(ax=ax, y_label='freq. choice', title="Choices")

    ax = axes[3]

    plot_mean_std(ax=ax, y=np.asarray(successes, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes",
              legend=False)

    if show:
        plt.tight_layout()
        plt.show()


POP_Q_VALUES, POP_P_CHOICES, POP_CHOICES, POP_SUCCESSES = \
    population_simulation(model=RL, param=PARAM_RL)

plot_pop_latent_variables(
    q_values=POP_Q_VALUES,
    p_choices=POP_P_CHOICES,
    choices=POP_CHOICES,
    successes=POP_SUCCESSES)


# ========================================================================
# Parameter fitting
# ========================================================================

class BanditOptimizer:

    def __init__(self,
                 choices,
                 successes,
                 model, bounds=None):
        self.choices = choices
        self.successes = successes
        self.model = model
        self.bounds = bounds

        self.t = 0

    def objective(self, param):

        n_iteration = len(self.choices)
        if isinstance(param, type(None)):
            assert self.model == Random
            agent = self.model(n_option=N)
        else:
            agent = self.model(n_option=N, *list(param))

        log_likelihood = np.zeros(n_iteration)

        # Simulate the task
        for t in range(n_iteration):
            choice, success = self.choices[t], self.successes[t]

            ps = agent.decision_rule()
            p_choice = ps[choice]

            log_likelihood[t] = np.log(p_choice + EPS)

            # Make agent learn
            agent.learn(option=choice, success=success)

        lls = np.sum(log_likelihood)
        v = -lls
        return v

    def _func(self, param):
        return self.objective(param)

    def run(self):

        if self.bounds:
            res = scipy.optimize.minimize(
                fun=self._func,
                x0=np.full(len(self.bounds), 0.5),
                bounds=self.bounds)
            assert res.success

            best_param = res.x
            best_value = res.fun

        else:
            assert self.model == Random
            best_param = None
            best_value = self.objective(None)

        return best_param, best_value


# ==========================================================================
# Simulation with best-fit parameters
# ==========================================================================


@use_pickle
def get_best_param():

    # Get best fit parameters
    opt = BanditOptimizer(
        choices=CHOICES,
        successes=SUCCESSES,
        model=RL,
        bounds=RL.bounds
    )

    best_param, best_value = opt.run()
    return best_param


BEST_PARAM = get_best_param()
print(f"Best-fit parameters: {BEST_PARAM}")


@use_pickle
def data_comparison_best_fit():

    # Get latent variables for best_fit
    q_values_bf_same_hist, p_choices_bf_same_hist = \
        data_latent_variables(
            model=RL,
            choices=CHOICES,
            successes=SUCCESSES,
            param=BEST_PARAM
        )

    # Run new simulation with best param
    choices_new, successes_new = \
        run_simulation(agent_model=RL,
                       param=BEST_PARAM, force=True)

    q_values_new, p_choices_new = data_latent_variables(
        model=RL,
        choices=choices_new,
        successes=successes_new,
        param=BEST_PARAM
    )

    return q_values_bf_same_hist, \
        p_choices_bf_same_hist, \
        q_values_new, \
        p_choices_new, \
        choices_new, \
        successes_new


def plot_comparison_best_fit(
        q_values_bf_same_hist,
        p_choices_bf_same_hist,
        q_values_new,
        p_choices_new,
        choices_new,
        successes_new
):

    n_cols = 3
    n_rows = 6
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(4*n_cols,  2.5*n_rows))

    titles = {
        "Initial": axes[0, 0],
        "Best-fit - Same hist.": axes[0, 1],
        "Best-fit - NEW hist.": axes[0, 2]
    }
    for title, ax in titles.items():
        ax.text(0.5, 1.2, title,
                horizontalalignment='center',
                transform=ax.transAxes,
                size=15, weight='bold')

    # Here for comparison
    plot_latent_variables(
        q_values=Q_VALUES, p_choices=P_CHOICES,
        choices=CHOICES,
        successes=SUCCESSES,
        axes=axes[:, 0])

    # Same history but best fit parameters
    plot_latent_variables(
        q_values=q_values_bf_same_hist,
        p_choices=p_choices_bf_same_hist,
        choices=CHOICES,
        successes=SUCCESSES,
        axes=axes[:, 1])

    # New simulation with best fit parameters
    plot_latent_variables(
        q_values=q_values_new,
        p_choices=p_choices_new,
        choices=choices_new,
        successes=successes_new,
        axes=axes[:, 2])

    plt.tight_layout()
    plt.show()


plot_comparison_best_fit(*data_comparison_best_fit())


# Population --------------------------------------------------------------

def data_pop_comparison_best_fit():

    # Run simulations with INITIAL parameters
    data_init = \
        population_simulation(model=RL, param=PARAM_RL)

    # Run new simulation with BEST parameters
    data_best = \
        population_simulation(model=RL, param=BEST_PARAM)

    return data_init, data_best


def plot_pop_comparison_best_fit(data_init, data_best):

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
    plot_pop_latent_variables(
        q_values=q_values,
        p_choices=p_choices,
        choices=choices,
        successes=successes,
        axes=axes[:, 0])

    # Data with best fit parameters
    q_values, p_choices, choices, successes = data_best
    plot_pop_latent_variables(
        q_values=q_values,
        p_choices=p_choices,
        choices=choices,
        successes=successes,
        axes=axes[:, 1])

    plt.tight_layout()
    plt.show()


plot_pop_comparison_best_fit(*data_pop_comparison_best_fit())

# =========================================================================
# Local minima exploration ================================================
# =========================================================================

# Local minima: Get data --------------------------------------------------


@use_pickle
def data_local_minima(model, choices, successes, grid_size=20):

    assert len(model.param_labels) == 2

    ll = np.zeros((grid_size, grid_size))

    opt = BanditOptimizer(
        choices=choices,
        successes=successes,
        model=model,
        bounds=model.bounds
    )

    param0_grid = np.linspace(*model.bounds[0], grid_size)
    param1_grid = np.linspace(*model.bounds[1], grid_size)

    for i in tqdm(range(len(param0_grid))):
        for j in range(len(param1_grid)):

            param_to_use = (param0_grid[i], param1_grid[j])

            ll_obs = - opt.objective(param=param_to_use)
            ll[j, i] = ll_obs

    x, y, z = param0_grid, param1_grid, ll

    return x, y, z


local_minima = data_local_minima(
    model=RL,
    choices=CHOICES,
    successes=SUCCESSES)


# Local minima: Plot --------------------------------------------------------

def plot_local_minima(
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

    x_coordinates, y_coordinates = np.meshgrid(x, y)

    c = ax.contourf(x_coordinates, y_coordinates, z,
                    levels=n_levels, cmap='viridis')

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(c, cax=cax)
    cbar.ax.set_ylabel('Log likelihood')

    ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


plot_local_minima(
    local_minima, labels=RL.param_labels,
    title='Local minima exploration')


# ==========================================================================
# PARAMETER RECOVERY =======================================================
# ==========================================================================

@use_pickle
def data_param_recovery(model, n_sets=30):

    param_labels = model.param_labels
    n_param = len(param_labels)

    param = {
        k: np.zeros((2, n_sets)) for k in param_labels
    }

    for set_idx in tqdm(range(n_sets)):

        param_to_simulate = np.zeros(2)

        for param_idx in range(n_param):
            v = np.random.uniform(*model.bounds[param_idx])

            param[param_labels[param_idx]][0, set_idx] = v
            param_to_simulate[param_idx] = v

        sim_choices, sim_successes = \
            run_simulation(
                agent_model=model,
                param=param_to_simulate,
            )

        opt = BanditOptimizer(
            choices=sim_choices,
            successes=sim_successes,
            model=model,
            bounds=model.bounds
        )

        best_param, best_value = opt.run()

        for param_idx in range(n_param):
            param[param_labels[param_idx]][1, set_idx] = best_param[param_idx]

    return param


param_rcv = data_param_recovery(RL)


# Parameter recovery: Plot -------------------------------------------------

def plot_parameter_recovery(data, x_label=None, y_label=None):

    keys = sorted(data.keys())
    n_keys = len(keys)
    colors = [f'C{i}' for i in range(n_keys)]

    # Create fig
    fig, axes = plt.subplots(ncols=n_keys, figsize=(3*n_keys, 3))

    for i in range(n_keys):
        ax = axes[i]
        k = keys[i]

        title = k

        x, y = data[k]

        ax.scatter(x, y, alpha=0.5, color=colors[i])

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.set_title(title)
        ax.set_xticks((0, 0.5, 1))
        ax.set_yticks((0, 0.5, 1))

        ax.plot(range(2), linestyle="--", alpha=0.2, color="black", zorder=-10)

        ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


plot_parameter_recovery(param_rcv, x_label='Simulated', y_label='Recovered')


# Parameter recovery: Stats -------------------------------------------------

def correlation_recovery(data):

    keys = sorted(data.keys())
    n_keys = len(keys)

    for i in range(n_keys):
        k = keys[i]

        x, y = data[k]
        cor, p = scipy.stats.pearsonr(x, y)
        print(f"[{k}] cor={cor:.3f}, p{format_p(p)}")

    print()


correlation_recovery(param_rcv)


# ===========================================================================
# BIC
# ===========================================================================

def bic(ll, k, n_iteration):
    return -2 * ll + k * np.log(n_iteration)


def compute_bic_scores():

    for m in MODELS:

        opt = BanditOptimizer(
            choices=CHOICES,
            successes=SUCCESSES,
            model=m,
            bounds=m.bounds
        )

        best_param, best_value = opt.run()

        ll = -best_value

        bic_score = bic(ll, len(m.bounds), n_iteration=T)

        print(f"BIC {m.__name__} ={bic_score:.3f}")


compute_bic_scores()


# ============================================================================
# Confusion matrix ===========================================================
# ============================================================================

# Confusion matrix: Get data -------------------------------------------------


@use_pickle
def data_confusion_matrix(models, n_sets=10):

    n_models = len(models)

    confusion_matrix = np.zeros((n_models, n_models))

    for i in tqdm(range(n_models)):

        model_to_simulate = models[i]

        for _ in range(n_sets):

            param_to_simulate = []
            for b in model_to_simulate.bounds:
                param_to_simulate.append(np.random.uniform(*b))

            sim_choices, sim_successes = \
                run_simulation(
                    agent_model=model_to_simulate,
                    param=param_to_simulate)

            bic_scores = np.zeros(n_models)

            for j in range(n_models):

                model_to_fit = models[j]

                opt = BanditOptimizer(
                    choices=sim_choices,
                    successes=sim_successes,
                    model=model_to_fit,
                    bounds=model_to_fit.bounds
                )

                best_param, best_value = opt.run()

                ll = -best_value

                bs = bic(ll, k=len(model_to_fit.bounds), n_iteration=T)

                bic_scores[j] = bs

            min_ = np.min(bic_scores)

            idx_min = np.arange(n_models)[bic_scores == min_]
            confusion_matrix[i, idx_min] += 1/len(idx_min)

        confusion_matrix[i] /= np.sum(confusion_matrix[i])

    return confusion_matrix


conf_mt = data_confusion_matrix(models=MODELS, n_sets=30)


# Confusion matrix: Plot -----------------------------------------------------

def plot_confusion_matrix(
        data,
        x_label, y_label,
        x_labels=None, y_labels=None, title="Confusion matrix"):

    if x_labels is None:
        x_labels = np.arange(data.shape[1])
    if y_labels is None:
        y_labels = np.arange(data.shape[0])

    fig, ax = plt.subplots(figsize=(2.5*data.shape[1],
                                    2.5*data.shape[0]))
    im = ax.imshow(data, alpha=0.5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # ... and move them again
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            ax.text(j, i, f'{data[i, j]:.3f}',
                    ha="center", va="center", color="black")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


confusion_matrix_labels = [m.__name__ for m in MODELS]
plot_confusion_matrix(data=conf_mt, x_label="simulated model",
                      y_label="fit model",
                      x_labels=confusion_matrix_labels,
                      y_labels=confusion_matrix_labels)
