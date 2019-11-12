"""
Modeling of decision-making
"""

# =================================================================
# Import your modules =============================================
# =================================================================

import os
import pickle
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

# =================================================================
# Globals =========================================================
# =================================================================

np.random.seed(123)
EPS = np.finfo(np.float).eps

BKP_FOLDER = "bkp"
os.makedirs(BKP_FOLDER, exist_ok=True)

USE_PICKLE_FIT = True
USE_PICKLE_BEST_FIT_COMPARISON = True
USE_PICKLE_PARAM_RECOVERY = True
USE_PICKLE_LOCAL_MINIMA = True
USE_PICKLE_CONFUSION_MATRIX = True

# ======================================================================
# Design your task =====================================================
# ======================================================================

N = 2
P = np.array([0.25, 0.75])
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

    ax.legend(lines, [r"$\tau=" + f'{v}$' for v in param_values])

    plt.plot()


plot_learning_parameter()
plot_temperature()

# =================================================================
# First artificial experiment =====================================
# =================================================================

# Param for the artificial subject
PARAM_RL = (0.1, 0.1)


def run_simulation(
        agent_model, param, n_iteration, n_option, prob_dist):

    if param is not None:
        agent = agent_model(n_option=n_option, *param)
    else:
        agent = agent_model(n_option=n_option)

    choices = np.zeros(n_iteration, dtype=int)
    successes = np.zeros(n_iteration, dtype=bool)

    # Simulate the task
    for t in range(n_iteration):
        # Determine choice
        choice = agent.choose()

        # Determine success
        p_success = prob_dist[choice]
        success = np.random.choice(
            [0, 1],
            p=np.array([1 - p_success, p_success]))

        # Make agent learn
        agent.learn(option=choice, success=success)

        # Backup
        choices[t] = choice
        successes[t] = success

    return choices, successes


def first_artificial_experiment():

    params = None, PARAM_RL

    n_models = len(MODELS)

    choices = {}
    successes = {}

    # Simulate the task
    for idx_model in range(n_models):
        _m = MODELS[idx_model]

        choices[_m.__name__], successes[_m.__name__] \
            = run_simulation(
                agent_model=_m, param=params[idx_model],
                n_iteration=T, n_option=N, prob_dist=P)

    return choices, successes


HIST_CHOICES, HIST_SUCCESSES = first_artificial_experiment()


# Plot results of simulations

def scatter_binary_choices(ax, y, color, label):

    ax.scatter(range(len(y)), y, color=color,
               alpha=0.2, label=label)

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(-0.02, 1.02)


def curve_rolling_mean(ax, y, color, label, window=50):

    d = pd.Series(y).rolling(window)
    means = d.mean()
    ax.plot(means, color=color, label=label)

    sd = d.std()
    y1 = means + sd
    y2 = means - sd
    ax.fill_between(
        range(len(means)),
        y1=y1,
        y2=y2,
        alpha=0.2,
        color=color
    )
    ax.set_ylim(-0.02, 1.02)


def multi_plot(data, func,
               x_label="time",
               y_label="choice"):

    keys = sorted(data.keys())
    n_keys = len(keys)

    fig, axes = plt.subplots(ncols=n_keys, figsize=(3*n_keys, 3))

    colors = [f'C{i}' for i in range(n_keys)]

    for i in range(n_keys):

        k = keys[i]
        ax = axes[i]
        color = colors[i]

        y = data[k]

        func(ax=ax, y=y, color=color, label=k)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.legend()

    plt.tight_layout()
    plt.show()


# Begin by the more basic possible
multi_plot(data=HIST_CHOICES, func=scatter_binary_choices)

# ...then maybe you can do better
multi_plot(data=HIST_CHOICES, func=curve_rolling_mean)


# (Behavioral) Stats ==========================================================

def format_p(p, threshold=0.05):

    pf = f'={p:.3f}' if p >= 0.001 else '<0.001'
    pf += " *" if p <= threshold else " NS"
    return pf


def stats():

    n_models = len(MODELS)

    contingency_table = np.zeros((n_models, N))

    for i in range(n_models):

        m = MODELS[i]
        m_name = m.__name__
        choices = HIST_CHOICES[m_name]

        k, n = np.sum(choices), len(choices)

        obs = n-k, k

        ci_low, ci_upp = \
            statsmodels.stats.proportion.proportion_confint(count=k, nobs=n)

        print(f"Model: {m_name}")
        print(f"prop choose best= {k/n:.3f}, CI=[{ci_low:.3f}, {ci_upp:.3f}]")

        chi2, p = scipy.stats.chisquare(obs)
        print("Chi2 for equality of proportion")
        print(f"Chi2={chi2:.3f}, p{format_p(p)}")
        print()

        contingency_table[i] = obs

    chi2, p, dof, ex = scipy.stats.chi2_contingency(contingency_table,
                                                    correction=False)
    print("Chi2 for independence")
    print(f"Chi2={chi2:.3f}, p{format_p(p)}")
    print()


stats()

# =======================================================================
# Latent variable =======================================================
# =======================================================================


def get_latent_variables(choices, successes, model, param):

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


def latent_variable_plot(q_values, p_choices, choices, axes=None):

    if axes is None:
        n_rows = 4
        fig, axes = plt.subplots(nrows=n_rows, figsize=(4, 2.5*n_rows))
        show = True
    else:
        show = False

    # Plot values
    ax = axes[0]

    lines = ax.plot(q_values)
    ax.legend(lines, [f"option {i}" for i in range(N)])
    ax.set_title("Q-values")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.set_ylim(-0.02, 1.02)

    # Plot probablilities
    ax = axes[1]

    lines = ax.plot(p_choices)
    ax.legend(lines, [f"option {i}" for i in range(N)])

    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Probabilities")
    ax.set_xlabel("time")
    ax.set_ylabel("p")

    # Plot scatter
    ax = axes[2]

    scatter_binary_choices(ax=ax, y=choices, color="C0", label="")
    ax.set_xlabel("time")
    ax.set_ylabel("choice")
    ax.set_title("Choices")

    # Plot average
    ax = axes[3]

    curve_rolling_mean(ax=ax, y=choices, color="C0", label="")

    ax.set_title("Choices (average)")

    ax.set_xlabel("time")
    ax.set_ylabel("choice")

    if show:
        plt.tight_layout()
        plt.show()


Q_VALUES, P_CHOICES = get_latent_variables(
        choices=HIST_CHOICES['RL'],
        successes=HIST_SUCCESSES['RL'],
        model=RL,
        param=PARAM_RL
    )

latent_variable_plot(q_values=Q_VALUES, p_choices=P_CHOICES,
                     choices=HIST_CHOICES['RL'])


# ========================================================================
# Parameter fitting
# ========================================================================

class BanditOptimizer:

    def __init__(self,
                 n_option,
                 choices,
                 successes,
                 model, bounds=None):
        self.n_option = n_option
        self.choices = choices
        self.successes = successes
        self.model = model
        self.bounds = bounds

        self.t = 0

    def objective(self, param):

        n_iteration = len(self.choices)
        if isinstance(param, type(None)):
            assert self.model == Random
            agent = self.model(n_option=self.n_option)
        else:
            agent = self.model(n_option=self.n_option, *list(param))

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


def get_data_plot_comparison_best_fit():

    # Get best fit parameters
    opt = BanditOptimizer(
        n_option=N,
        choices=HIST_CHOICES['RL'],
        successes=HIST_SUCCESSES['RL'],
        model=RL,
        bounds=RL.bounds
    )

    best_param, best_value = opt.run()

    # Get latent variables for best_fit
    q_values_bf_same_hist, p_choices_bf_same_hist = \
        get_latent_variables(
            model=RL,
            choices=HIST_CHOICES['RL'],
            successes=HIST_SUCCESSES['RL'],
            param=best_param
        )

    # Run new simulation with best param
    choices_new, successes_new = \
        run_simulation(agent_model=RL,
                       param=best_param,
                       n_iteration=T,
                       n_option=N,
                       prob_dist=P)

    q_values_new, p_choices_new = get_latent_variables(
        model=RL,
        choices=choices_new,
        successes=successes_new,
        param=best_param
    )

    return best_param, \
        {
            "q_values_bf_same_hist": q_values_bf_same_hist,
            "p_choices_bf_same_hist": p_choices_bf_same_hist,
            "choices_new": choices_new,
            "q_values_new": q_values_new,
            "p_choices_new": p_choices_new
        }


bkp_file = os.path.join(BKP_FOLDER, "comparison_best_fit.p")
if not os.path.exists(bkp_file) or not USE_PICKLE_FIT:

    best_param, data_comparison_best_fit = \
        get_data_plot_comparison_best_fit()
    pickle.dump(
        (best_param, data_comparison_best_fit), open(bkp_file, 'wb'))

else:
    best_param, data_comparison_best_fit = pickle.load(open(bkp_file, 'rb'))


print(f"Best-fit parameters: {best_param}")


def plot_comparison_best_fit(
        q_values_bf_same_hist,
        p_choices_bf_same_hist,
        choices_new,
        q_values_new,
        p_choices_new
):

    n_cols = 3
    n_rows = 4
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
    latent_variable_plot(
        q_values=Q_VALUES, p_choices=P_CHOICES,
        choices=HIST_CHOICES['RL'],
        axes=axes[:, 0])

    # Same history but best fit parameters
    latent_variable_plot(
        q_values=q_values_bf_same_hist,
        p_choices=p_choices_bf_same_hist,
        choices=HIST_CHOICES['RL'],
        axes=axes[:, 1])

    # New simulation with best fit parameters
    latent_variable_plot(
        q_values=q_values_new,
        p_choices=p_choices_new,
        choices=choices_new,
        axes=axes[:, 2])

    plt.tight_layout()
    plt.show()

    # Full simulation with best fit parameters


plot_comparison_best_fit(**data_comparison_best_fit)


# =========================================================================
# Local minima exploration ================================================
# =========================================================================

# Local minima: Get data --------------------------------------------------

def local_minima(model, choices, successes, grid_size=20):

    assert len(model.param_labels) == 2

    ll = np.zeros((grid_size, grid_size))

    opt = BanditOptimizer(
        n_option=N,
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


bkp_file = os.path.join(BKP_FOLDER, "local_minima.p")

if not os.path.exists(bkp_file) or not USE_PICKLE_LOCAL_MINIMA:

    data_local_minima = local_minima(
        model=RL,
        choices=HIST_CHOICES['RL'],
        successes=HIST_SUCCESSES['RL'])
    pickle.dump(data_local_minima, open(bkp_file, 'wb'))

else:
    data_local_minima = pickle.load(open(bkp_file, 'rb'))


# Local minima: Plot --------------------------------------------------------

def phase_diagram(
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


phase_diagram(data_local_minima, labels=RL.param_labels,
              title='Local minima exploration')


# ==========================================================================
# PARAMETER RECOVERY =======================================================
# ==========================================================================

def param_recovery(model, n_sets=30):

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
                n_iteration=T,
                n_option=N,
                prob_dist=P
            )

        opt = BanditOptimizer(
            n_option=N,
            choices=sim_choices,
            successes=sim_successes,
            model=model,
            bounds=model.bounds
        )

        best_param, best_value = opt.run()

        for param_idx in range(n_param):
            param[param_labels[param_idx]][1, set_idx] = best_param[param_idx]

    return param


bkp_file = os.path.join(BKP_FOLDER, "param_recovery.p")

if not os.path.exists(bkp_file) or not USE_PICKLE_PARAM_RECOVERY:
    param_rcv = param_recovery(RL)
    pickle.dump(param_rcv, open(bkp_file, 'wb'))

else:
    param_rcv = pickle.load(open(bkp_file, 'rb'))


# Parameter recovery: Plot -------------------------------------------------

def scatter(data, x_label=None, y_label=None):

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


scatter(param_rcv, x_label='Simulated', y_label='Recovered')


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

    choices = HIST_CHOICES['RL']
    successes = HIST_SUCCESSES['RL']

    for m in MODELS:

        opt = BanditOptimizer(
            n_option=N,
            choices=choices,
            successes=successes,
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

def data_for_confusion_matrix(models, n_sets=10):

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
                    param=param_to_simulate,
                    n_iteration=T,
                    n_option=N,
                    prob_dist=P)

            bic_scores = np.zeros(n_models)

            for j in range(n_models):

                model_to_fit = models[j]

                opt = BanditOptimizer(
                    n_option=N,
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


bkp_file = os.path.join(BKP_FOLDER, "conf_mt.p")


if not os.path.exists(bkp_file) or not USE_PICKLE_CONFUSION_MATRIX:
    conf_mt = data_for_confusion_matrix(models=MODELS)
    pickle.dump(conf_mt, open(bkp_file, 'wb'))

else:
    conf_mt = pickle.load(open(bkp_file, 'rb'))


# Confusion matrix: Plot -----------------------------------------------------

def confusion_matrix_plot(
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
confusion_matrix_plot(data=conf_mt, x_label="simulated model",
                      y_label="fit model",
                      x_labels=confusion_matrix_labels,
                      y_labels=confusion_matrix_labels)
