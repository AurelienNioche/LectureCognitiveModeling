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
import scipy.stats
from tqdm.autonotebook import tqdm

np.random.seed(123)
EPS = np.finfo(np.float).eps

# %%

N = 2
P = np.array([0.5, 0.75])
T = 200


# %%

class Random:
    """
    No-learning model
    """

    def __init__(self, n_option):
        self.n_option = n_option
        self.options = np.arange(n_option)

    def choose(self):
        p = self.decision_rule()
        return np.random.choice(self.options, p=p)

    def learn(self, i, success):
        self.updating_rule(i=i, success=success)

    def decision_rule(self):
        return np.ones(self.n_option) / self.n_option

    def updating_rule(self, i, success):
        pass


# %%

class RL(Random):
    """
    Reinforcement learning model
    """

    def __init__(self, learning_rate, temp, n_option, initial_value=0.5):
        super().__init__(n_option=n_option)
        self.values = np.full(n_option, initial_value)
        self.learning_rate = learning_rate
        self.temp = temp

    def decision_rule(self):

        p_soft = np.exp(self.values / self.temp) / \
             np.sum(np.exp(self.values / self.temp), axis=0)
        return p_soft

    def updating_rule(self, i, success):
        self.values[i] += self.learning_rate * (success - self.values[i])


# # %% md
#
# # 3. Simulate
#
# # %%
#
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
        agent.learn(i=choice, success=success)

        # Backup
        choices[t] = choice
        successes[t] = success

    return choices, successes
#
#
# # %%
#
# np.random.seed(0)
#
# learning_rate, temp = 0.01, 0.1
#
# models = Random, RL
# params = None, (learning_rate, temp)
#
# n_models = len(models)
#
# hist_choices = {}
# hist_successes = {}
#
# # Simulate the task
# for i in range(n_models):
#     m = models[i]
#
#     choices, successes = run_simulation(
#         agent_model=m, param=params[i],
#         n_iteration=T, n_option=N, prob_dist=P)
#
#     hist_choices[m.__name__] = choices
#     hist_successes[m.__name__] = successes
#
#
# # %%
#
# def basic_scatter(data, y_label="choice", x_label="time"):
#     keys = sorted(data.keys())
#     n_keys = len(keys)
#
#     fig, axes = plt.subplots(ncols=n_keys, figsize=(10, 4))
#
#     colors = [f'C{i}' for i in range(n_keys)]
#
#     for i in range(n_keys):
#         k = keys[i]
#         y = data[k]
#
#         ax = axes[i]
#         ax.scatter(range(len(y)), y, color=colors[i],
#                    alpha=0.2, label=k)
#
#         ax.set_ylim(-0.02, 1.02)
#         ax.yaxis.set_major_locator(MaxNLocator(integer=True))
#
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         ax.legend()
#
#     plt.tight_layout()
#     plt.show()
#
#
# basic_scatter(data=hist_choices)
#
#
# # %%
#
# def running_mean(data, y_label='choice', x_label="time", window=20):
#     keys = sorted(data.keys())
#     n_keys = len(keys)
#
#     fig, axes = plt.subplots(ncols=n_keys)
#
#     colors = [f'C{i}' for i in range(n_keys)]
#
#     for i in range(n_keys):
#         k = keys[i]
#
#         y = data[k]
#
#         ax = axes[i]
#         ax.plot(pd.Series(y).rolling(window).mean(),
#                 color=colors[i], alpha=0.2, label=k)
#         ax.set_ylim(-0.02, 1.02)
#
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#
#         ax.legend()
#     plt.tight_layout()
#     plt.show()
#
#
# running_mean(data=hist_choices)
#
# # %%
#
# import statsmodels.stats.proportion
# import scipy.stats
#
# for i in range(n_models):
#     m = models[i]
#     choices = hist_choices[m.__name__]
#
#     k, n = np.sum(choices), len(choices)
#     print(statsmodels.stats.proportion.proportion_confint(count=k, nobs=n))
#     freq_yes = k / n
#     freq_no = 1 - freq_yes
#     chi2, p = scipy.stats.chisquare((k, n - k))
#     print(chi2, p)
#
#
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
        agent = self.model(n_option=self.n_option, *param)

        log_likelihood = np.zeros(n_iteration)

        # Simulate the task
        for t in range(n_iteration):
            choice, success = self.choices[t], self.successes[t]

            ps = agent.decision_rule()
            p_choice = ps[choice]

            log_likelihood[t] = np.log(p_choice + EPS)

            # Make agent learn
            agent.learn(i=choice, success=success)

        lls = np.sum(log_likelihood)
        v = -lls

        return v

    def _func(self, param):
        return self.objective(param)

    def run(self):
        res = scipy.optimize.minimize(
            fun=self._func,
            x0=np.full(len(self.bounds), 0.5),
            bounds=self.bounds)
        assert res.success

        best_param = res.x
        best_value = res.fun

        return best_param, best_value

# %%


def param_recovery(
        param_labels=("alpha", "beta"),
        bounds=((0, 1), (0.01, 1))):
    n_sets = 30

    n_param = len(param_labels)

    param = {
        k: np.zeros((2, n_sets)) for k in param_labels
    }

    for set_idx in tqdm(range(n_sets)):

        param_to_simulate = np.zeros(2)

        for param_idx in range(n_param):
            v = np.random.uniform(*bounds[param_idx])

            param[param_labels[param_idx]][0, set_idx] = v
            param_to_simulate[param_idx] = v

        sim_choices, sim_successes = \
            run_simulation(
                agent_model=RL,
                param=param_to_simulate,
                n_iteration=T,
                n_option=N,
                prob_dist=P
            )

        opt = BanditOptimizer(
            n_option=N,
            choices=sim_choices,
            successes=sim_successes,
            model=RL,
            bounds=bounds
        )

        best_param, best_value = opt.run()

        for param_idx in range(n_param):
            param[param_labels[param_idx]][1, set_idx] = best_param[param_idx]

    return param


# %%

BKP_FOLDER = "bkp"
os.makedirs(BKP_FOLDER, exist_ok=True)
bkp_file = os.path.join(BKP_FOLDER, "param_recovery.p")
force = True

if not os.path.exists(bkp_file) or force:
    param_rcv = param_recovery()
    pickle.dump(param_rcv, open(bkp_file, 'wb'))

else:
    param_rcv = pickle.load(open(bkp_file, 'rb'))


# %%

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

        # max_ = max(x+y)
        # min_ = min(x+y)
        #
        # ax.set_xlim(min_, max_)
        # ax.set_ylim(min_, max_)

        # ticks_positions = [round(j, 2) for j in np.linspace(min_, max_, 3)]

        # ax.set_xticks(ticks_positions)
        # ax.set_yticks(ticks_positions)

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
        print(f"[{k}] cor={cor}, p={p:.3f}")

# %%



# %%

def bic(ll, k, n_iteration):
    return -2 * ll + k * np.log(n_iteration)
