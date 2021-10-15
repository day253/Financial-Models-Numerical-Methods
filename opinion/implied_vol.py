#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functions.Parameters import Option_param
from functions.Processes import Diffusion_process, Merton_process, VG_process, Heston_process
from functions.BS_pricer import BS_pricer
from functions.Merton_pricer import Merton_pricer
from functions.VG_pricer import VG_pricer
from functions.Heston_pricer import Heston_pricer

import numpy as np
import pandas as pd
import scipy as scp
import scipy.stats as ss
import matplotlib.pyplot as plt
import scipy.optimize as scpo
from functools import partial
from itertools import compress
import os
import warnings
warnings.filterwarnings("ignore")

BS = BS_pricer


def implied_volatility(price, S0, K, T, r, diff, payoff="call", method="fsolve", disp=True):
    """ Returns Implied volatility
        methods:  fsolve (default) or brent
    """

    def obj_fun(vol):
        return (price - BS.BlackScholes(payoff=payoff, S0=S0, K=K, T=T, r=r, diff=diff, sigma=vol))

    if method == "brent":
        x, r = scpo.brentq(obj_fun, a=1e-15, b=500, full_output=True)
        if r.converged == True:
            return x
    if method == "fsolve":
        X0 = [0.1, 0.5, 1, 3]   # set of initial guess points
        for x0 in X0:
            x, _, solved, _ = scpo.fsolve(
                obj_fun, x0, full_output=True, xtol=1e-8)
            if solved == 1:
                return x[0]

    if disp == True:
        print("Strike", K)
    return -1


def implied_vol_minimize(price, S0, K, T, r, diff, payoff="call", disp=True):
    """ Returns Implied volatility by minimization"""

    n = 2     # must be even

    def obj_fun(vol):
        return (BS.BlackScholes(payoff=payoff, S0=S0, K=K, T=T, r=r, diff=diff, sigma=vol) - price)**n

    res = scpo.minimize_scalar(obj_fun, bounds=(1e-15, 8), method='bounded')
    if res.success == True:
        return res.x
    if disp == True:
        print("Strike", K)
    return -1
