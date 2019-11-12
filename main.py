"""
Modeling of decision-making
"""


# %% md

# 0. Import the necessary libraries

# %%

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
import itertools as it


np.random.seed(123)
EPS = np.finfo(np.float).eps

# %%

N = 2
P = np.array([0.25, 0.75])
T = 500


# %%

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


# %%

class RL(Random):
    """
    Reinforcement learning model
    """
    param_labels = ("alpha", "beta")
    bounds = (0.01, 1), (0.05, 0.1)

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


# %% md


def plot_learning_parameter():

    param_values = (0.1, 0.25, 0.5)
    n_iteration = 100

    n_param_values = len(param_values)

    values = np.zeros((n_iteration, n_param_values))

    for i in range(n_param_values):
        alpha = param_values[i]
        agent = RL(learning_rate=alpha,
                   temp=0.0, n_option=1)
        for t in range(n_iteration):

            values[t, i] = agent.values[0]
            agent.learn(option=0, success=1)

    fig, ax = plt.subplots()
    lines = ax.plot(values)

    ax.set_xlabel("time")
    ax.set_ylabel("value")

    ax.legend(lines, [r"\alpha=" + f'{v}' for v in param_values])

    plt.plot()


def plot_temperature():

    param_values = (0.05, 0.5, 1)

    x_values = np.linspace(-1, 1, 100)

    n_param_values = len(param_values)

    values = np.zeros((len(x_values), n_param_values))

    for i in range(n_param_values):
        alpha = param_values[i]


    fig, ax = plt.subplots()
    ax.plot(values)

    ax.set_xlabel("Q(A) - Q(B)")
    ax.set_ylabel("p(A)")

    plt.plot()



# 3. Simulate

# %%

def run_simulation(agent_model, param, n_iteration, n_option,
                   prob_dist):

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
        success = np.random.choice([0, 1],
                                   p=np.array([1 - p_success, p_success]))

        # Make agent learn
        agent.learn(option=choice, success=success)

        # Backup
        choices[t] = choice
        successes[t] = success

    return choices, successes


# # %%

PARAM_RL = (0.1, 0.1)

MODELS = Random, RL


params = None, PARAM_RL

n_models = len(MODELS)

hist_choices = {}
hist_successes = {}

# Simulate the task
for idx_model in range(n_models):
    _m = MODELS[idx_model]

    hist_choices[_m.__name__], hist_successes[_m.__name__] \
        = run_simulation(
            agent_model=_m, param=params[idx_model],
            n_iteration=T, n_option=N, prob_dist=P)


# # %%

def scatter_binary_choices(ax, y, color, label):

    ax.scatter(range(len(y)), y, color=color,
               alpha=0.2, label=label)

    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


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


def multi_plot(data, func, x_label="time", y_label="choice"):

    keys = sorted(data.keys())
    n_keys = len(keys)

    fig, axes = plt.subplots(ncols=n_keys, figsize=(10, 4))

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


# Basic scatterplot
multi_plot(data=hist_choices, func=scatter_binary_choices)

# Running mean
multi_plot(data=hist_choices, func=curve_rolling_mean)
#
# # %%


def format_p(p, threshold=0.05):
    pf = f'={p:.3f}' if p >= 0.001 else '<0.001'
    pf += " *" if p <= threshold else " NS"
    return pf


def stats():

    contingency_table = np.zeros((n_models, N))

    for i in range(n_models):

        m = MODELS[i]
        m_name = m.__name__
        choices = hist_choices[m_name]

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
    # chi2, p = scipy.stats.chisquare((k, n - k))
    print("Chi2 for independence")
    print(f"Chi2={chi2:.3f}, p{format_p(p)}")
    print()


stats()


def data_for_latent_variable_plot(
        param, n_option, n_iteration,
        prob_dist):

    agent = RL(n_option=n_option, *param)

    choices = np.zeros(n_iteration, dtype=int)
    successes = np.zeros(n_iteration, dtype=bool)
    values = np.zeros((n_iteration, n_option))
    p_choices = np.zeros((n_iteration, n_option))

    # Simulate the task
    for t in range(n_iteration):

        # Register choice
        values[t] = agent.values

        # Get choice
        p_choices[t] = agent.decision_rule()

        # Determine choice
        choice = np.random.choice(np.arange(n_option), p=p_choices[t])

        # Determine success
        p_success = prob_dist[choice]
        success = np.random.choice(
            [0, 1],
            p=[1 - p_success, p_success])

        # Make agent learn
        agent.learn(option=choice, success=success)

        # Backup
        choices[t] = choice
        successes[t] = success

    fig, axes = plt.subplots(nrows=4)

    # Plot values
    ax = axes[0]

    lines = ax.plot(values)
    ax.legend(lines, [f"option {i}" for i in range(n_option)])
    ax.set_title("Q-values")
    ax.set_xlabel("time")
    ax.set_ylabel("value")

    # Plot probablilities
    ax = axes[1]

    lines = ax.plot(values)
    ax.legend(lines, [f"option {i}" for i in range(n_option)])
    ax.set_title("Probabilities")
    ax.set_xlabel("time")
    ax.set_ylabel("p")

    # Plot scatter
    ax = axes[2]

    scatter_binary_choices(ax=ax, y=choices, color="C0", label="")

    # plot choices
    ax = axes[3]

    curve_rolling_mean(ax=ax, y=choices, color="C0", label="")

    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("time")
    ax.set_ylabel("choice")

    return choices, successes


# # %%


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

# %%


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


# %%

BKP_FOLDER = "bkp"
os.makedirs(BKP_FOLDER, exist_ok=True)
bkp_file = os.path.join(BKP_FOLDER, "param_recovery.p")
force = False

if not os.path.exists(bkp_file) or force:
    param_rcv = param_recovery(RL)
    pickle.dump(param_rcv, open(bkp_file, 'wb'))

else:
    param_rcv = pickle.load(open(bkp_file, 'rb'))


# %%


def phase_diagram(
        data,
        labels,
        n_levels=100,
        fontsize=10,
        title=None):

    x, y, z = data
    x_label, y_label = labels

    fig, ax = plt.subplots(figsize=(8, 8))

    # Axes labels
    ax.set_xlabel(x_label, fontsize=fontsize * 1.5)
    ax.set_ylabel(y_label, fontsize=fontsize * 1.5)

    # Title
    ax.set_title(title)

    X, Y = np.meshgrid(x, y)

    c = ax.contourf(X, Y, z, levels=n_levels, cmap='viridis')

    divider = make_axes_locatable(ax)

    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = plt.colorbar(c, cax=cax)
    cbar.ax.set_ylabel('Log likelihood')

    ax.set_aspect(1)

    plt.show()


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

if not os.path.exists(bkp_file) or force:

    data_local_minima = local_minima(
        model=RL,
        choices=hist_choices['RL'],
        successes=hist_successes['RL'])
    pickle.dump(data_local_minima, open(bkp_file, 'wb'))

else:
    data_local_minima = pickle.load(open(bkp_file, 'rb'))

phase_diagram(data_local_minima, labels=RL.param_labels)
# confusion_matrix(data=data_local_minima[-1],
#                  x_labels=data_local_minima[0],
#                  y_labels=data_local_minima[1])


def scatter(data):

    keys = sorted(data.keys())
    n_keys = len(keys)
    colors = [f'C{i}' for i in range(n_keys)]

    # Create fig
    fig, axes = plt.subplots(ncols=n_keys, figsize=(5, 10))

    for i in range(n_keys):
        ax = axes[i]
        k = keys[i]

        title = k

        x, y = data[k]

        ax.scatter(x, y, alpha=0.5, color=colors[i])

        ax.set_title(title)
        ax.set_xticks((0, 0.5, 1))
        ax.set_yticks((0, 0.5, 1))

        ax.plot(range(2), linestyle="--", alpha=0.2, color="black", zorder=-10)

        ax.set_aspect(1)

    plt.tight_layout()
    plt.show()


# %%

scatter(param_rcv)

# %%


def correlation_recovery(data):

    keys = sorted(data.keys())
    n_keys = len(keys)

    for i in range(n_keys):
        k = keys[i]

        x, y = data[k]
        cor, p = scipy.stats.pearsonr(x, y)
        print(f"[{k}] cor={cor:.3f}, p{format_p(p)}")

    print()


# %%

correlation_recovery(param_rcv)

# %%


def bic(ll, k, n_iteration):
    return -2 * ll + k * np.log(n_iteration)


def compute_bic_scores():
    opt = BanditOptimizer(
        n_option=N,
        choices=hist_choices['RL'],
        successes=hist_successes['RL'],
        model=RL,
        bounds=RL.bounds
    )

    best_param, best_value = opt.run()

    ll = -best_value

    bic_score = bic(ll, len(RL.bounds), n_iteration=T)

    print(f"BIC RL REVOVERED={bic_score:.3f}")

    print(best_param, ll)

    opt = BanditOptimizer(
        n_option=N,
        choices=hist_choices['RL'],
        successes=hist_successes['RL'],
        model=Random,
        bounds=None
    )

    ll = - opt.objective(param=None)

    bic_score = bic(ll, 0, n_iteration=T)

    print(f"BIC Random={bic_score:.3f}")


compute_bic_scores()


def confusion_matrix_plot(data,
                          x_label, y_label,
                          x_labels=None, y_labels=None, title=None):

    if x_labels is None:
        x_labels = np.arange(data.shape[1])
    if y_labels is None:
        y_labels = np.arange(data.shape[0])

    fig, ax = plt.subplots(figsize=(8, 8))
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
    # divider = make_axes_locatable(ax)
    # cax = divider.append_axes("right", size="5%", pad=0.05)

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    # import matplotlib as mpl

    # cbar = fig.colorbar(im) #, cax=cax)  # , ticks=y_ticks)
    # cbar.ax.set_ylabel('Log likelihood')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)

    # import matplotlib as mpl

    cbar = plt.colorbar(im, cax=cax)  # , ticks=y_ticks)
    # cbar.ax.set_ylabel('Log likelihood')

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


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
force = False

if not os.path.exists(bkp_file) or force:
    conf_mt = data_for_confusion_matrix(models=MODELS)
    pickle.dump(conf_mt, open(bkp_file, 'wb'))

else:
    conf_mt = pickle.load(open(bkp_file, 'rb'))


labels = [m.__name__ for m in MODELS]
confusion_matrix_plot(data=conf_mt, x_label="simulated model",
                      y_label="fit model", x_labels=labels, y_labels=labels)
