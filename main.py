"""
Modeling of decision-making
"""

# =================================================================
# Import your modules =============================================
# =================================================================

import numpy as np
import scipy.optimize
import scipy.stats
from tqdm.autonotebook import tqdm

from utils.decorator import use_pickle
import stats.stats as stats
import plot.plot as plot

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

N_SUBJECTS = 30

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


# =================================================================
# Define your model space =========================================
# =================================================================

MODELS = Random, WSLS, RW, RWCK
MODEL_NAMES = [m.__name__ for m in MODELS]

# =================================================================
# Study the effect of your parameters =============================
# =================================================================

# For RW only...
plot.learning_rate(model=RW)
plot.softmax_temperature()


# =================================================================
# Single agent simulation =========================================
# =================================================================

@use_pickle
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

    return choices, successes


# Get data ----------------------------------------------------
SEED_XP = 0
MODEL_XP = RW
PARAM_XP = (0.1, 0.1)
CHOICES_XP, SUCCESSES_XP = \
    run_simulation(agent_model=RW, param=PARAM_XP, seed=SEED_XP)

# Plot --------------------------------------------------------
# Begin by the more basic possible
plot.behavior_basic(choices=CHOICES_XP, successes=SUCCESSES_XP)

# ...then maybe you can do better
plot.behavior_average(choices=CHOICES_XP, successes=SUCCESSES_XP)


# =======================================================================
# Latent variable =======================================================
# =======================================================================

def latent_variables_rw(choices, successes, param):

    """
    Specific to RW
    """

    # Create the agent
    agent = RW(*param)

    # Data containers
    q_values = np.zeros((T, N))
    p_choices = np.zeros((T, N))

    # (Re-)Simulate the task
    for t in range(T):

        # Register values
        q_values[t] = agent.q_values

        # Register probabilities of choices
        p_choices[t] = agent.decision_rule()

        # Make agent learn
        agent.learn(option=choices[t],
                    success=successes[t])

    return q_values, p_choices


# Get the data
Q_VALUES, P_CHOICES = latent_variables_rw(choices=CHOICES_XP,
                                          successes=SUCCESSES_XP,
                                          param=PARAM_XP)

# Plot
plot.latent_variables_rw(q_values=Q_VALUES, p_choices=P_CHOICES,
                         successes=SUCCESSES_XP, choices=CHOICES_XP)


# ========================================================================
# Population simulation
# ========================================================================

@use_pickle
def run_sim_pop_rw(model, param, n_subjects):

    """
    Specific to RW
    """

    # Data containers
    pop_q_values = np.zeros((n_subjects, T, N))
    pop_p_choices = np.zeros((n_subjects, T, N))
    pop_choices = np.zeros((n_subjects, T), dtype=int)
    pop_successes = np.zeros((n_subjects, T), dtype=bool)

    for i in range(n_subjects):

        # Get choices and successes
        choices, successes \
            = run_simulation(seed=i, agent_model=model, param=param)

        # Get q-values and choice probabilities
        q_values, p_choices \
            = latent_variables_rw(
                choices=choices, successes=successes,
                param=param)

        # Backup
        pop_q_values[i] = q_values
        pop_p_choices[i] = p_choices
        pop_choices[i] = choices
        pop_successes[i] = successes

    return pop_q_values, pop_p_choices, pop_choices, pop_successes


# Get the data
POP_Q_VALUES, POP_P_CHOICES, POP_CHOICES, POP_SUCCESSES = \
    run_sim_pop_rw(model=RW, param=PARAM_XP, n_subjects=N_SUBJECTS)

# Plot
plot.pop_latent_variables_rw(
    q_values=POP_Q_VALUES, p_choices=POP_P_CHOICES,
    choices=POP_CHOICES, successes=POP_SUCCESSES)


# ========================================================================
# Parameter optimization
# ========================================================================

class BanditOptimizer:

    """
    Given a series of choices and successes, and a DM model,
    estimate the best-fit param
    """

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

        log_likelihood_sum = np.sum(log_likelihood)
        v = -log_likelihood_sum
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

    # Create optimizer
    opt = BanditOptimizer(
        choices=CHOICES_XP,
        successes=SUCCESSES_XP,
        model=RW
    )

    # Run the optimization
    best_param, best_value = opt.run()
    return best_param


# Get the best-fit parameters
BEST_PARAM = get_best_param()
print(f"'True' parameters: {PARAM_XP}")
print(f"Best-fit parameters: {tuple(BEST_PARAM)}\n")


# New simulation with best-fit parameters
CHOICES_BF, SUCCESSES_BF = \
    run_simulation(seed=SEED_XP + 1, agent_model=RW, param=BEST_PARAM)

# Get the values of the latent variables
Q_VALUES_BF, P_CHOICES_BF = \
    latent_variables_rw(choices=CHOICES_BF, successes=SUCCESSES_BF,
                        param=BEST_PARAM)

# Plot
plot.comparison_best_fit_rw(
    q_values=Q_VALUES, p_choices=P_CHOICES,
    choices=CHOICES_XP, successes=SUCCESSES_XP,
    choices_bf=CHOICES_BF, successes_bf=SUCCESSES_BF,
    q_values_bf=Q_VALUES_BF, p_choices_bf=P_CHOICES_BF)


# Population --------------------------------------------------------------

def data_pop_comparison_best_fit():

    # Run simulations with INITIAL parameters
    data_init = \
        run_sim_pop_rw(model=RW, param=PARAM_XP, n_subjects=N_SUBJECTS)

    # Run new simulation with BEST parameters
    data_best = \
        run_sim_pop_rw(model=RW, param=BEST_PARAM, n_subjects=N_SUBJECTS)

    return data_init, data_best


# Get data
DATA_INIT, DATA_BEST = data_pop_comparison_best_fit()

# Plot
plot.pop_comparison_best_fit(data_init=DATA_INIT, data_best=DATA_BEST)


# =========================================================================
# parameter space exploration =============================================
# =========================================================================

@use_pickle
def parameter_space_exploration(model, choices, successes, grid_size=20):

    """
    Compute likelihood for several combinations of parameters
    (using grid exploration)
    :param model: DM model with 2 free paramters
    :param choices: array-like
    :param successes: array-like
    :param grid_size: int
    :return: tuple of three vectors
    """

    assert len(model.param_labels) == 2, \
        "this function is designed for models that have " \
        "at least and at most 2 parameters"
    assert hasattr(model, 'fit_bounds'), \
        f"{model.__name__} has not 'fit_bounds' attribute"

    # Container for log-likelihood
    ll = np.zeros((grid_size, grid_size))

    # Create the optimizer
    opt = BanditOptimizer(choices=choices,
                          successes=successes,
                          model=model)

    # Create a grid for each parameter
    param0_grid = np.linspace(*model.fit_bounds[0], grid_size)
    param1_grid = np.linspace(*model.fit_bounds[1], grid_size)

    # Loop over each value of the parameter grid for both parameters
    for i in tqdm(range(len(param0_grid))):
        for j in range(len(param1_grid)):

            # Select the parameter to use
            param_to_use = (param0_grid[i], param1_grid[j])

            # Call the objective function of the optimizer
            ll_obs = - opt.objective(param=param_to_use)

            # Backup
            ll[j, i] = ll_obs

    # Return three vectors:
    # x: values of first parameter
    # y: values of second parameter
    # z: likelihood given specific combination of parameters
    x, y, z = param0_grid, param1_grid, ll

    return x, y, z


# Get data
PARAM_SPACE_EXPLO = parameter_space_exploration(
    model=RW,
    choices=CHOICES_XP,
    successes=SUCCESSES_XP)

# Plot
plot.parameter_space_exploration(
    PARAM_SPACE_EXPLO,
    labels=RW.param_labels,
    title='Parameter space exploration')


# ==========================================================================
# PARAMETER RECOVERY =======================================================
# ==========================================================================

@use_pickle
def data_param_recovery(model, n_sets):

    print("Computing data for parameter recovery...")

    # Get the parameters labels
    param_labels = model.param_labels
    n_param = len(param_labels)

    # Data container (2: simulated, retrieved)
    param = {
        k: np.zeros((2, n_sets)) for k in param_labels
    }

    # Loop over the number of parameter sets
    for set_idx in tqdm(range(n_sets)):

        # Select parameter to simulate...
        param_to_sim = \
            [np.random.uniform(*b)
             for b in model.fit_bounds]

        # Simulate
        choices, successes = run_simulation(seed=set_idx,
                                            agent_model=model,
                                            param=param_to_sim)

        # Create the optimizer and run it
        opt = BanditOptimizer(choices=choices,
                              successes=successes,
                              model=model)
        best_param, best_value = opt.run()

        # Backup
        for i in range(n_param):
            param[param_labels[i]][0, set_idx] = param_to_sim[i]
            param[param_labels[i]][1, set_idx] = best_param[i]

    return param


# Get data
P_RCV = data_param_recovery(model=RW, n_sets=30)

# Plot
plot.parameter_recovery(data=P_RCV)

# Stats
stats.correlation_recovery(data=P_RCV)


# ===========================================================================
# Model comparison
# ===========================================================================

def bic(ll, k, n_iteration):
    return -2 * ll + k * np.log(n_iteration)


def optimize_and_compare(choices, successes):

    n_models = len(MODELS)
    bic_scores = np.zeros(n_models)
    lls = np.zeros(n_models)
    best_params = []

    for j in range(n_models):

        # Select the model
        model_to_fit = MODELS[j]

        # Create the optimizer and run it
        opt = BanditOptimizer(choices=choices,
                              successes=successes,
                              model=model_to_fit)
        best_param, best_value = opt.run()

        # Get log-likelihood for best param
        ll = -best_value

        # Compute the bit score
        bs = bic(ll, k=len(model_to_fit.fit_bounds), n_iteration=T)

        # Backup
        bic_scores[j] = bs
        lls[j] = ll
        best_params.append(best_param)

    return best_params, lls, bic_scores


@use_pickle
def comparison_single_subject():

    best_params, lls, bic_scores = \
        optimize_and_compare(choices=CHOICES_XP, successes=SUCCESSES_XP)

    print(f"Model used: {MODEL_XP.__name__}")
    for i, m in enumerate(MODELS):

        print(f"BIC {m.__name__} = {bic_scores[i]:.3f}")


# Compute bic scores for evey model for our initial set of data
comparison_single_subject()


# ============================================================================
# Confusion matrix ===========================================================
# ============================================================================

@use_pickle
def data_confusion_matrix(models, n_sets):

    print("Computing data for confusion matrix...")

    # Number of models
    n_models = len(models)

    # Data container
    confusion_matrix = np.zeros((n_models, n_models))

    # Loop over each model
    for i in tqdm(range(n_models)):

        # Select the model
        model_to_sim = models[i]

        for j in range(n_sets):

            # Select parameters to simulate
            param_to_sim = \
                [np.random.uniform(*b)
                 for b in model_to_sim.fit_bounds]

            # Simulate
            choices, successes = \
                run_simulation(
                    seed=j,
                    agent_model=model_to_sim,
                    param=param_to_sim)

            # Compute bic scores
            best_params, lls, bic_scores = \
                optimize_and_compare(choices=choices, successes=successes)

            # Get minimum value for bic (min => best)
            min_ = np.min(bic_scores)

            # Get index of models that get best bic
            idx_min = np.arange(n_models)[bic_scores == min_]

            # Add result in matrix
            confusion_matrix[i, idx_min] += 1/len(idx_min)

    return confusion_matrix


# Data
N_SETS_CONF = 30
CONF_MT = data_confusion_matrix(models=MODELS, n_sets=N_SETS_CONF)

# Plot
plot.confusion_matrix(data=CONF_MT, tick_labels=MODEL_NAMES)

# Stats
stats.classification(CONF_MT, model_names=MODEL_NAMES)


# ======================================================================
# Fake experiment  =====================================================
# ======================================================================

@use_pickle
def data_fake_xp(model, n_subjects):

    assert hasattr(model, 'fit_bounds'), \
        f"{model.__name__} has not 'fit_bounds' attribute"

    # Data containers
    choices = np.zeros((n_subjects, T), dtype=int)
    successes = np.zeros((n_subjects, T), dtype=bool)
    bic_scores = np.zeros((n_subjects, len(MODELS)))
    lls = np.zeros((n_subjects, len(MODELS)))
    bp = np.zeros(n_subjects, dtype=object)

    # Loop over subjects
    for i in tqdm(range(n_subjects)):

        # Select parameters (limited range)
        param = [np.random.uniform(*b) for b in model.xp_bounds]

        # Simulate
        choices[i], successes[i] = \
            run_simulation(seed=i, agent_model=model, param=param)

        # Optimize and compare
        bp[i], lls[i], bic_scores[i] = \
            optimize_and_compare(choices=choices[i], successes=successes[i])

    # Freq and confidence intervals for the barplot
    lls_freq, lls_err = stats.freq_and_err(lls)
    bic_freq, bic_err = stats.freq_and_err(-bic_scores)

    # Look at the best model
    best_model_idx = int(np.argmax(bic_freq))

    # Assume that it should be the one that you used to simulate
    assert best_model_idx == MODELS.index(model)

    # Simulate with best parameters
    choices_bf = np.zeros((n_subjects, T), dtype=int)
    successes_bf = np.zeros((n_subjects, T), dtype=bool)

    for i in tqdm(range(n_subjects)):

        param = bp[i][best_model_idx]
        choices_bf[i], successes_bf[i] = run_simulation(
            seed=i+1,
            agent_model=model, param=param)

    return {
        "choices": choices,
        "successes": successes,
        "lls": lls,
        "bic_scores": bic_scores,
        "bic_freq": bic_freq,
        "bic_err": bic_err,
        "lls_freq": lls_freq,
        "lls_err": lls_err,
        "choices_bf": choices_bf,
        "successes_bf": successes_bf}


# Get _data
DATA_XP = data_fake_xp(model=MODEL_XP, n_subjects=N_SUBJECTS)

# Plot
plot.fake_xp(model_names=MODEL_NAMES, n_option=N, **DATA_XP)
