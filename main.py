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
    Random selection
    """
    param_labels = ()
    fit_bounds = ()
    xp_bounds = ()

    def __init__(self):
        self.options = np.arange(N)

    def choose(self):
        p = self.decision_rule()
        return np.random.choice(self.options, p=p)

    def learn(self, option, success):
        self.updating_rule(option=option, success=success)

    def decision_rule(self):
        return np.ones(N) / N

    def updating_rule(self, option, success):
        pass


class WSLS(Random):

    """
    Win-Stay-Lose-Switch
    """

    param_labels = ("epsilon", )
    fit_bounds = (0., 1),
    xp_bounds = (0.25, 0.75),

    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

        self.c = -1
        self.r = -1

    def decision_rule(self):

        if self.c == -1:
            return np.ones(N) / N  # First turn

        p = np.zeros(N)

        # 1 - epsilon: select the same
        # epsilon: choose randomly
        # ...so p apply rule
        p_apply_rule = 1 - self.epsilon
        p_random = self.epsilon / N
        if self.r:
            p[self.c] = p_apply_rule + p_random
            p[self.options != self.c] = p_random
        else:
            p[self.options != self.c] = p_apply_rule + p_random
            p[self.c] = p_random

        return p

    def updating_rule(self, option, success):
        self.r = success
        self.c = option


class RW(Random):
    """
    Rescola-Wagner
    """
    param_labels = ("alpha", "beta")
    fit_bounds = (0.01, 1), (0.05, 1)
    xp_bounds = (0.1, 0.25), (0.01, 0.15)

    def __init__(self, q_learning_rate, q_temp, initial_value=0.5):
        super().__init__()
        self.q_values = np.full(N, initial_value)
        self.q_learning_rate = q_learning_rate
        self.q_temp = q_temp

    def decision_rule(self):

        p_soft = np.exp(self.q_values / self.q_temp) / \
                 np.sum(np.exp(self.q_values / self.q_temp))
        return p_soft

    def updating_rule(self, option, success):
        self.q_values[option] += \
            self.q_learning_rate * (success - self.q_values[option])


class RWCK(RW):

    """
    Rescola-Wagner / Choice-Kernel
    """

    param_labels = ("alpha", "beta", "alpha_c", "beta_c")
    fit_bounds = (0.01, 1), (0.05, 1), (0.01, 1), (0.05, 1)
    xp_bounds = (0.1, 0.3), (0.05, 0.15), (0.1, 0.3), (0.05, 0.15)

    def __init__(self, q_learning_rate, q_temp, c_learning_rate, c_temp):

        super().__init__(q_learning_rate=q_learning_rate, q_temp=q_temp)
        self.c_learning_rate = c_learning_rate
        self.c_temp = c_temp
        self.c_values = np.zeros(N)

    def decision_rule(self):

        p_soft = np.exp(
            (self.q_values / self.q_temp) +
            (self.c_values / self.c_temp)
        ) / \
             np.sum(np.exp(
                     (self.q_values / self.q_temp) +
                     (self.c_values / self.c_temp)
             ))
        return p_soft

    def updating_rule(self, option, success):

        a = np.zeros(N, dtype=int)
        a[option] = 1
        self.c_values[:] += \
            self.c_learning_rate * (a - self.c_values[:])

        super().updating_rule(option=option, success=success)


MODELS = Random, WSLS, RW, RWCK

# Study the effect of your parameters ========================================


def plot_learning_parameter(n_iteration=100,
                            param_values=(0.01, 0.1, 0.2, 0.3)):

    n_param_values = len(param_values)

    values = np.zeros((n_iteration, n_param_values))

    for i in range(n_param_values):
        alpha = param_values[i]
        agent = RW(q_learning_rate=alpha,
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


@use_pickle
def run_simulation(seed, agent_model, param=()):

    np.random.seed(seed)

    agent = agent_model(*param)

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
    run_simulation(agent_model=RW, param=PARAM_RL, seed=SEED)


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

    ax.scatter(range(len(y)), y+np.random.uniform(-0.2, 0.2, size=len(y)),
               color=colors[y],
               alpha=0.2, s=10)

    for i, color in enumerate(colors):
        ax.scatter(-1, -1, color=color, alpha=0.2,
                   label=scatter_labels[i],
                   s=20)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xticks((0, int(T/2), T))
    ax.set_yticks((0, 1))
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(-0.02*T, T*1.02)
    ax.set_xlabel("time")
    ax.set_ylabel(y_label)
    ax.set_aspect(50)

    if title is not None:
        ax.set_title(title)


    ax.legend()


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

    """
    This function aims to work only with RescolaWagner
    :param choices:
    :param successes:
    :param model:
    :param param:
    :return:
    """

    agent = model(*param)

    assert hasattr(agent, "q_values"), \
        "Model instance needs to have 'values' attribute"

    q_values = np.zeros((T, N))
    p_choices = np.zeros((T, N))

    # (Re-)Simulate the task
    for t in range(T):

        # Register values
        q_values[t] = agent.q_values

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
        model=RW,
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
            = run_simulation(seed=i, agent_model=model, param=param)

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
    population_simulation(model=RW, param=PARAM_RL)

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
                 model):
        self.choices = choices
        self.successes = successes
        self.model = model

        assert hasattr(model, 'fit_bounds'), \
            f"{model.__name__} has not 'fit_bounds' attribute"

        self.t = 0

    def objective(self, param):

        n_iteration = len(self.choices)
        if isinstance(param, type(None)):
            assert self.model == Random
            agent = self.model()
        else:
            agent = self.model(*param)

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

        if self.model.fit_bounds:
            res = scipy.optimize.minimize(
                fun=self._func,
                x0=np.full(len(self.model.fit_bounds), 0.5),
                bounds=self.model.fit_bounds)
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
        model=RW
    )

    best_param, best_value = opt.run()
    return best_param


BEST_PARAM = get_best_param()
print(f"Best-fit parameters: {BEST_PARAM}")


@use_pickle
def data_comparison_best_fit():

    # # Get latent variables for best_fit
    # q_values_bf_same_hist, p_choices_bf_same_hist = \
    #     data_latent_variables(
    #         model=RW,
    #         choices=CHOICES,
    #         successes=SUCCESSES,
    #         param=BEST_PARAM
    #     )

    # Run new simulation with best param
    new_seed = SEED + 1
    choices_new, successes_new = \
        run_simulation(seed=new_seed,
                       agent_model=RW,
                       param=BEST_PARAM)

    q_values_new, p_choices_new = data_latent_variables(
        model=RW,
        choices=choices_new,
        successes=successes_new,
        param=BEST_PARAM
    )

    return \
        q_values_new, \
        p_choices_new, \
        choices_new, \
        successes_new


def plot_comparison_best_fit(
        q_values_new,
        p_choices_new,
        choices_new,
        successes_new
):

    n_cols = 2
    n_rows = 6
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols,
                             figsize=(4*n_cols,  2.5*n_rows))

    titles = {
        "Initial": axes[0, 0],
        "Best-fit": axes[0, 1]
        # "Best-fit - Same hist.": axes[0, 1],
        # "Best-fit - NEW hist.": axes[0, 2]
    }
    for title, ax in titles.items():
        ax.text(0.5, 1.2, title,
                horizontalalignment='center',
                transform=ax.transAxes,
                size=15, weight='bold')

    # Here for comparison
    plot_latent_variables(
        q_values=Q_VALUES,
        p_choices=P_CHOICES,
        choices=CHOICES,
        successes=SUCCESSES,
        axes=axes[:, 0])

    # # Same history but best fit parameters
    # plot_latent_variables(
    #     q_values=q_values_bf_same_hist,
    #     p_choices=p_choices_bf_same_hist,
    #     choices=CHOICES,
    #     successes=SUCCESSES,
    #     axes=axes[:, 1])

    # New simulation with best fit parameters
    plot_latent_variables(
        q_values=q_values_new,
        p_choices=p_choices_new,
        choices=choices_new,
        successes=successes_new,
        axes=axes[:, 1])

    plt.tight_layout()
    plt.show()


plot_comparison_best_fit(*data_comparison_best_fit())


# Population --------------------------------------------------------------

def data_pop_comparison_best_fit():

    # Run simulations with INITIAL parameters
    data_init = \
        population_simulation(model=RW, param=PARAM_RL)

    # Run new simulation with BEST parameters
    data_best = \
        population_simulation(model=RW, param=BEST_PARAM)

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
    assert hasattr(model, 'fit_bounds'), \
        f"{model.__name__} has not 'fit_bounds' attribute"

    ll = np.zeros((grid_size, grid_size))

    opt = BanditOptimizer(
        choices=choices,
        successes=successes,
        model=model
    )

    param0_grid = np.linspace(*model.fit_bounds[0], grid_size)
    param1_grid = np.linspace(*model.fit_bounds[1], grid_size)

    for i in tqdm(range(len(param0_grid))):
        for j in range(len(param1_grid)):

            param_to_use = (param0_grid[i], param1_grid[j])

            ll_obs = - opt.objective(param=param_to_use)
            ll[j, i] = ll_obs

    x, y, z = param0_grid, param1_grid, ll

    return x, y, z


local_minima = data_local_minima(
    model=RW,
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
    local_minima, labels=RW.param_labels,
    title='Local minima exploration')


# ==========================================================================
# PARAMETER RECOVERY =======================================================
# ==========================================================================

@use_pickle
def data_param_recovery(model, n_sets):

    param_labels = model.param_labels
    n_param = len(param_labels)

    param = {
        k: np.zeros((2, n_sets)) for k in param_labels
    }

    for set_idx in tqdm(range(n_sets)):

        param_to_simulate = np.zeros(2)

        for param_idx in range(n_param):
            v = np.random.uniform(*model.fit_bounds[param_idx])

            param[param_labels[param_idx]][0, set_idx] = v
            param_to_simulate[param_idx] = v

        sim_choices, sim_successes = \
            run_simulation(
                seed=set_idx,
                agent_model=model,
                param=param_to_simulate,
            )

        opt = BanditOptimizer(
            choices=sim_choices,
            successes=sim_successes,
            model=model
        )

        best_param, best_value = opt.run()

        for param_idx in range(n_param):
            param[param_labels[param_idx]][1, set_idx] = best_param[param_idx]

    return param


param_rcv = data_param_recovery(model=RW, n_sets=30)


# Parameter recovery: Plot -------------------------------------------------

def plot_parameter_recovery(data):

    x_label = 'Simulated'
    y_label = 'Recovered'

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


plot_parameter_recovery(param_rcv)


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


def compute_bic_scores(choices, successes):

    n_models = len(MODELS)
    bic_scores = np.zeros(n_models)
    lls = np.zeros(n_models)
    best_params = []

    for j in range(n_models):
        model_to_fit = MODELS[j]

        opt = BanditOptimizer(
            choices=choices,
            successes=successes,
            model=model_to_fit
        )

        best_param, best_value = opt.run()
        ll = -best_value

        bs = bic(ll, k=len(model_to_fit.fit_bounds), n_iteration=T)

        bic_scores[j] = bs
        lls[j] = ll
        best_params.append(best_param)

    return best_params, lls, bic_scores


@use_pickle
def first_computation_bic():

    best_params, lls, bic_scores = \
        compute_bic_scores(choices=CHOICES, successes=SUCCESSES)
    for i, m in enumerate(MODELS):

        print(f"BIC {m.__name__} = {bic_scores[i]:.3f}")


first_computation_bic()


# ============================================================================
# Confusion matrix ===========================================================
# ============================================================================

# Confusion matrix: Get data -------------------------------------------------


@use_pickle
def data_confusion_matrix(models, n_sets):

    n_models = len(models)

    confusion_matrix = np.zeros((n_models, n_models))

    for i in tqdm(range(n_models)):

        model_to_simulate = models[i]

        for j in range(n_sets):

            param_to_simulate = \
                [np.random.uniform(*b)
                 for b in model_to_simulate.fit_bounds]

            sim_choices, sim_successes = \
                run_simulation(
                    seed=j,
                    agent_model=model_to_simulate,
                    param=param_to_simulate)

            best_params, lls, bic_scores = compute_bic_scores(
                choices=sim_choices,
                successes=sim_successes)

            min_ = np.min(bic_scores)

            idx_min = np.arange(n_models)[bic_scores == min_]
            confusion_matrix[i, idx_min] += 1/len(idx_min)

    return confusion_matrix


N_SETS_CONF = 30
conf_mt = data_confusion_matrix(models=MODELS, n_sets=N_SETS_CONF)


# Confusion matrix: Stats -----------------------------------------------------

def stats_conf_mt(obs):

    print("STATS CONFUSION MATRIX")

    for i in range(len(obs)):

        f_obs = obs[i]
        # tot = np.sum(data[i])
        # f_exp = [tot if j == i else 0 for j in range(len(data[i]))]
        chi2, p = scipy.stats.chisquare(f_obs=f_obs)

        print(f"Chi2={chi2:.3f}, p{format_p(p)}")

        k = obs[i, i]
        n = np.sum(obs[i])
        ci_low, ci_upp = \
            statsmodels.stats.proportion.proportion_confint(count=k, nobs=n)
        print(f"prop choose best= {k/n:.3f}, CI=[{ci_low:.3f}, {ci_upp:.3f}]")
    #
    # chi2, p, dof, expctd = \
    #     scipy.stats.chi2_contingency(obs, correction=False)
    #
    # print(f"Chi2 Cont={chi2:.3f}, p{format_p(p)}")
    # print("expctd", expctd)


stats_conf_mt(conf_mt)


# Confusion matrix: Plot -----------------------------------------------------

def plot_confusion_matrix(data):

    # Normalize
    for i in range(len(data)):
        data[i] /= np.sum(data[i])

    confusion_matrix_labels = [m.__name__ for m in MODELS]

    title = "Confusion matrix"
    x_label = "simulated model"
    y_label = "fit model"
    x_tick_labels = confusion_matrix_labels
    y_tick_labels = confusion_matrix_labels

    fig, ax = plt.subplots(figsize=(2*data.shape[1],
                                    2*data.shape[0]))
    im = ax.imshow(data, alpha=0.5)

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
            ax.text(j, i, f'{data[i, j]:.3f}',
                    ha="center", va="center", color="black")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

    ax.set_title(title)

    fig.tight_layout()
    plt.show()


plot_confusion_matrix(data=conf_mt)

# ======================================================================
# Fake experiment  =====================================================
# ======================================================================


@use_pickle
def data_fake_xp(model, n_subjects):

    assert hasattr(model, 'fit_bounds'), \
        f"{model.__name__} has not 'fit_bounds' attribute"

    choices = np.zeros((n_subjects, T), dtype=int)
    successes = np.zeros((n_subjects, T), dtype=bool)
    bic_scores = np.zeros((n_subjects, len(MODELS)))
    lls = np.zeros((n_subjects, len(MODELS)))
    best_params = []

    for i in tqdm(range(n_subjects)):

        param = [np.random.uniform(*b) for b in model.xp_bounds]
        c, s = run_simulation(seed=i,
                              agent_model=model, param=param)

        choices[i] = c
        successes[i] = s

        bp, lls[i], bic_scores[i] = \
            compute_bic_scores(choices=c, successes=s)

        best_params.append(bp)

    freq, y_err = freq_and_err(-bic_scores)

    best_model_idx = int(np.argmax(freq))

    assert best_model_idx == MODELS.index(model)

    new_choices = np.zeros((n_subjects, T), dtype=int)
    new_successes = np.zeros((n_subjects, T), dtype=bool)

    for i in tqdm(range(n_subjects)):

        param = best_params[i][best_model_idx]
        new_choices[i], new_successes[i] = run_simulation(
            seed=i+1,
            agent_model=model, param=param)

    return \
        choices, successes, \
        lls, bic_scores, \
        new_choices, new_successes


def plot_model_metric(metrics, ax, y_label, title):

    n = len(MODELS)

    # Colors
    colors = np.array([f"C{i}" for i in range(n)])

    # For boxplot
    positions = list(range(n))
    values_box_plot = [[] for _ in range(n)]

    x_scatter = []
    y_scatter = []
    colors_scatter = []

    x_tick_labels = [m.__name__ for m in MODELS]

    for i in range(n):

        for v in metrics[:, i]:

            # For boxplot
            values_box_plot[i].append(v)

            # For scatter
            x_scatter.append(i + np.random.uniform(-0.05*n, 0.05*n))
            y_scatter.append(v)
            colors_scatter.append(colors[i])

    ax.scatter(x_scatter, y_scatter, c=colors_scatter, s=20, alpha=0.2,
               linewidth=0.0, zorder=1)

    ax.set_ylabel(y_label)
    ax.set_title(title)

    # ax.invert_yaxis()

    # Boxplot
    bp = ax.boxplot(values_box_plot, positions=positions,
                    labels=x_tick_labels, showfliers=False, zorder=2)

    for e in ['boxes', 'caps', 'whiskers', 'medians']:
        for b in bp[e]:
            b.set(color='black')


def freq_and_err(data):

    n_subjects, n_models = data.shape

    counts = np.zeros(n_models)

    max_user = np.max(data, axis=1)

    for i in range(n_models):
        did_best = data[:, i] == max_user
        counts[i] = np.sum(did_best)

    ci = np.zeros((2, n_models))

    for i in range(n_models):
        ci[:, i] = \
            statsmodels.stats.proportion.proportion_confint(
                count=counts[i], nobs=n_subjects)

    freq = counts / n_subjects

    y_err = np.absolute(np.subtract(ci, freq))
    return freq, y_err


def plot_bar_best_metric(ax, freq, y_err, y_label, title):

    x_tick_labels = [m.__name__ for m in MODELS]

    x_pos = np.arange(len(freq))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_tick_labels)

    ax.bar(x_pos, freq, yerr=y_err)

    for i in range(len(freq)):

        ax.text(x=i, y=freq[i]+y_err[1, i]+0.05,
                s=f'{freq[i]:.2f}', size=6,
                horizontalalignment='center',
                )

    ax.set_yticks((0, 0.5, 1))
    ax.set_ylim((0, max(1, np.max(freq[:]+y_err[1, :])+0.15)))

    ax.set_title(title)
    ax.set_ylabel(y_label)


def plot_fake_xp(choices, successes, lls, bic_scores,
                 new_choices, new_successes):

    n_rows = 8
    fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5 * n_rows))

    # Plot average
    ax = axes[0]

    for i in range(N):
        y = choices == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')

    custom_ax(ax=ax, y_label='freq. choice', title="Choices")

    ax = axes[1]

    plot_mean_std(ax=ax, y=np.asarray(successes, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes",
              legend=False)

    ax = axes[2]

    plot_model_metric(ax=ax, metrics=lls,
                      title="Log-Likelihood Sums",
                      y_label="LLS")

    ax = axes[3]
    freq, y_err = freq_and_err(lls)
    plot_bar_best_metric(ax=ax, freq=freq, y_err=y_err,
                         y_label='Highest LLS (freq.)',
                         title="Highest LLS")

    ax = axes[4]
    plot_model_metric(ax=ax, metrics=bic_scores,
                      title="BIC Scores",
                      y_label="BIC")
    ax.invert_yaxis()

    ax = axes[5]
    freq, y_err = freq_and_err(-bic_scores)
    plot_bar_best_metric(ax=ax, freq=freq, y_err=y_err,
                         y_label='Lowest BIC (freq.)',
                         title="Lowest BIC")

    # Plot average
    ax = axes[6]

    for i in range(N):
        y = new_choices == i
        plot_mean_std(ax=ax, y=np.asarray(y, dtype=int), label=f'option {i}')

    custom_ax(ax=ax, y_label='freq. choice', title="Choices Best-fit")

    ax = axes[7]

    plot_mean_std(ax=ax, y=np.asarray(new_successes, dtype=int))
    custom_ax(ax=ax, y_label='freq. success', title="Successes Best-fit",
              legend=False)

    plt.tight_layout()
    plt.show()


def fake_xp():

    model_to_simulate = RW
    choices, successes, lls, bic_scores, new_choices, new_successes = \
        data_fake_xp(model=model_to_simulate, n_subjects=30)
    plot_fake_xp(choices=choices,
                 successes=successes,
                 new_choices=new_choices,
                 new_successes=new_successes,
                 bic_scores=bic_scores,
                 lls=lls)
    # u, p = scipy.stats.mannwhitneyu(bic_scores[:, 1], bic_scores[:, 2])
    # print([f"{i:.3f}" for i in bic_scores[:, 1]])
    # print([f"{i:.3f}" for i in bic_scores[:, 2]])
    # print(u, p)


fake_xp()
