# def first_artificial_experiment():
#
#     params = None, PARAM_RL
#
#     n_models = len(MODELS)
#
#     choices = {}
#     successes = {}
#
#     # Simulate the task
#     for idx_model in range(n_models):
#         _m = MODELS[idx_model]
#
#     choices[_m.__name__], successes[_m.__name__] \
#         = run_simulation(
#             agent_model=_m, param=params[idx_model],
#             n_iteration=T, n_option=N, prob_dist=P)
#
#     return choices, successes