import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

N = 2
P = np.array([0.5, 0.75])
T = 500

N_SUBJECTS = 30

# Get data
SEED_SINGLE = 0
PARAM_SINGLE = np.array([0.10, 10.00])


class GenericPlayer:
    """
    Generic Player
    """

    param_labels = ()
    fit_bounds = ()

    def __init__(self):
        self.options = np.arange(N)

    def choose(self):
        p = self.decision_rule()
        return np.random.choice(self.options, p=p)

    def learn(self, option, success):
        self.updating_rule(option=option, success=success)

    def decision_rule(self):
        raise NotImplementedError

    def updating_rule(self, option, success):
        pass


class RW(GenericPlayer):
    """
    Rescorla-Wagner
    """

    param_labels = r"$\alpha$", r"$\beta$",
    fit_bounds = (0.0, 1.0), (1.0, 20.0),

    def __init__(self, q_alpha, q_beta, initial_value=0.5):
        super().__init__()
        self.q_values = np.full(N, initial_value)
        self.q_alpha = q_alpha
        self.q_beta = q_beta

    def decision_rule(self):
        p_soft = np.exp(self.q_beta * self.q_values) / \
                 np.sum(np.exp(self.q_beta * self.q_values))
        return p_soft

    def updating_rule(self, option, success):
        self.q_values[option] += \
            self.q_alpha * (success - self.q_values[option])


def run_sim_pop(model, param, n_subjects, seed):

    print(f"Running simulation for {n_subjects} agents...")

    # Data containers
    frames = []

    for i in tqdm(range(n_subjects), file=sys.stdout):

        # Get choices and successes
        single_d = run_simulation(seed=seed[i], agent_model=model, param=param[i])
        single_d["id"] = seed[i]
        frames.append(single_d)

    return pd.concat(frames)


def run_simulation(seed, agent_model, param=()):

    # Seed the pseudo-random number generator
    np.random.seed(seed)

    # Create the agent
    agent = agent_model(*param)

    # Data containers
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

    return pd.DataFrame({"choice": choices,
                         "success": successes,
                         "time": np.arange(T)})

# We will experiment with Rescola-Wagner
MODEL_XP = RW

PARAM_HOM_POP = [PARAM_SINGLE for _ in range(N_SUBJECTS)]

# hom_pop_bhv = run_sim_pop(model=MODEL_XP,
#                           param=PARAM_HOM_POP,
#                           n_subjects=N_SUBJECTS,
#                           seed=np.arange(N_SUBJECTS))

# hom_pop_bhv.to_csv("tamere.csv", index=False)

hom_pop_bhv = pd.read_csv("tamere.csv")


groups = hom_pop_bhv.groupby(['time', 'choice']).size().to_frame('size').reset_index()
groups.reset_index(inplace=True)

print("groups")
print(groups)

sns.lineplot(data=groups, x="time", y="size", hue="choice")
plt.show()


hom_pop_bhv["choice"] = hom_pop_bhv['choice'].apply(str)
sns.lineplot(data=hom_pop_bhv, x='time', y='choice')
plt.show()

# # sns.relplot(data=hom_pop_bhv, kind="line")
#
# data = sns.load_dataset("flights")
# print(data.head())
#
# print(data.pivot("year", "month", "passengers"))

# sns.lineplot(data=groups.size(), x="time", y=groups.size().index, hue="choice")
# plt.show()

# n = len(hom_pop_bhv.id.unique())
# print("n", n)
# print(a[0].groupby("time").count())
# print()
