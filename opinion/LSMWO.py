#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy as scp
import scipy.stats as ss


def LSMWO(S0, K, T, r, diff, sig, payoff, ExpValue, ExpTime, N=2000, paths=30000, order=2):
    """
    Longstaff-Schwartz Method for pricing American options

    N = number of time steps
    paths = number of generated paths
    order = order of the polynomial for the regression
    """

    dt = T/(N-1)          # time interval
    df = np.exp(-r * dt)  # discount factor per time time interval

    X0 = np.zeros((paths, 1))
    increments = ss.norm.rvs(loc=(diff-sig**2/2)*dt,
                             scale=np.sqrt(dt)*sig, size=(paths, N-1))
    X = np.concatenate((X0, increments), axis=1).cumsum(1)
    S = S0 * np.exp(X)
    ExpT = np.int(np.floor(ExpTime*N/T))  # 预期时间

    if payoff == "call":
        S = S[S[:, :ExpT].max(1) > ExpValue]  # 筛选符合要求的路径
        H = np.maximum(S - K, 0)
    else:
        S = S[S[:, :ExpT].min(1) < ExpValue]
        H = np.maximum(K - S, 0)    # intrinsic values for put option
    V = np.zeros_like(H)            # value matrix
    V[:, -1] = H[:, -1]

    # Valuation by LS Method
    for t in range(N-2, 0, -1):
        good_paths = H[:, t] > 0
        if any(good_paths) > 0:
            rg = np.polyfit(S[good_paths, t], V[good_paths, t+1]
                            * df, 2)    # polynomial regression
            # evaluation of regression
            C = np.polyval(rg, S[good_paths, t])

            exercise = np.zeros(len(good_paths), dtype=bool)
            exercise[good_paths] = H[good_paths, t] > C

            V[exercise, t] = H[exercise, t]
            V[exercise, t+1:] = 0
            discount_path = (V[:, t] == 0)
            V[discount_path, t] = V[discount_path, t+1] * df
        else:
            V[:, t] = V[:, t+1] * df

    V0 = np.mean(V[:, 1]) * df  #
    return V0
