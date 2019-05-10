import numpy as np
from scipy.optimize import fsolve


def a_cr(T_stag, k, R):
    return (2 * k * R * T_stag / (k + 1))**0.5


def tau_lam(lam, k):
    """ГДФ температруы через приведенную скорость"""
    return 1 - (k - 1) / (k + 1) * lam**2


def pi_lam(lam, k):
    """ГДФ давления через приведенную скорость"""
    return (1 - (k - 1) / (k + 1) * lam**2) ** (k / (k - 1))


def eps_lam(lam, k):
    """ГДФ плотности через приведенную скорость"""
    return (1 - (k - 1) / (k + 1) * lam**2) ** (1 / (k - 1))


def tau_M(M, k):
    """ГДФ температуры через число Маха"""
    return 1 + (k - 1) / 2 * M ** 2


def pi_M(M, k):
    """ГДФ давления через число Маха"""
    return (1 + (k - 1) / 2 * M ** 2) ** (k / (k - 1))


def lam(k, **kwargs):
    if 'tau' in kwargs:
        return np.sqrt((1 - kwargs['tau']) * (k + 1) / (k - 1))
    if 'pi' in kwargs:
        return np.sqrt((k + 1) / (k - 1) * (1 - kwargs['pi']**((k - 1) / k)))
    if 'q' and 'kind' in kwargs:
        q0 = kwargs['q']
        kind = kwargs['kind']
        if kind == 'subs':
            return fsolve(lambda x: [q(x[0], k) - q0], np.array([0.5]))[0]
        if kind == 'supers':
            return fsolve(lambda x: [q(x[0], k) - q0], np.array([1.5]))[0]


def q(lam, k):
        return ((k + 1) / 2) ** (1 / (k - 1)) * lam * (1 - (k - 1) / (k + 1) * lam**2) ** (1 / (k - 1))


def m(k):
    return (2 / (k + 1)) ** ((k + 1) / (2 * (k - 1))) * k ** 0.5
