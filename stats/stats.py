import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats.proportion


def rolling_mean(y, window=50):
    """
    Compute the rolling mean
    :param y: array-like
    :param window: float
    :return: array-like
    """
    return pd.Series(y).rolling(window).mean()


def confidence_interval(k, n):

    """
    Compute the confidence interval at 95%
    :param k: int
    :param n: int
    :return: tuple of float
    """

    return statsmodels.stats.proportion.proportion_confint(count=k, nobs=n)


def format_p(p, threshold=0.05):

    """
    Format the display of the p-value
    :param p: float
    :param threshold: float
    :return: string
    """
    pf = f'={p:.3f}' if p >= 0.001 else '<0.001'
    pf += " *" if p <= threshold else " NS"
    return pf


def correlation_recovery(data):

    print("\n" + "=" * 10)
    print("\nStatistical analysis for the parameter recovery")
    print("=" * 10 + "\n")

    keys = sorted(data.keys())
    n_keys = len(keys)

    for i in range(n_keys):
        k = keys[i]

        x, y = data[k]
        cor, p = scipy.stats.pearsonr(x, y)
        print(f"[{k}] cor={cor:.3f}, p{format_p(p)}")

    print()


def freq_and_err(data):

    n_subjects, n_models = data.shape

    counts = np.zeros(n_models)

    max_user = np.max(data, axis=1)

    for i in range(n_models):
        did_best = data[:, i] == max_user
        counts[i] = np.sum(did_best)

    ci = np.zeros((2, n_models))

    for i in range(n_models):
        ci[:, i] = confidence_interval(counts[i], n_subjects)

    freq = counts / n_subjects

    y_err = np.absolute(np.subtract(ci, freq))
    return freq, y_err


def classification(obs, model_names):
    print("\n" + "=" * 10)
    print("Statistic analysis for the classification")
    print("=" * 10 + "\n")

    for i in range(len(obs)):

        print(f"{model_names[i]}")
        print("-" * 10)

        k = obs[i, i]
        n = np.sum(obs[i])
        precision = k/n
        ci_low, ci_upp = confidence_interval(k, n)
        print(f'k{k}, n{n}')
        print(f"Precision: {precision:.3f}, CI=[{ci_low:.3f}, {ci_upp:.3f}]")

        n = np.sum(obs[:, i])
        recall = k/n
        ci_low, ci_upp = confidence_interval(k, n)
        print(f'k{k}, n{n}')
        print(f"Recall: {recall:.3f}, CI=[{ci_low:.3f}, {ci_upp:.3f}]")
        print(f"F1 score = {2*(precision* recall)/(precision+recall):.3f}")
        print()
