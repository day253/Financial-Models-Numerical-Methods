#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 最好锁定随机数种子，应该可以提速

import numpy as np
import pandas as pd
import scipy as scp
import scipy.stats as ss

from implied_vol import implied_vol_minimize
from LSMWO import LSMWO
from utility import GetRunTime


@GetRunTime
def Choice_opinion(stockname, ExpValue, ExpTime):

    # opinions 初始输入参数：股票名称，预期股价，预期时间
    stockname = stockname
    ExpValue = ExpValue
    ExpTime = ExpTime/365  # 获取的期权合约的剩余期限应大于预期时间

    # 取期权截面数据
    filename = "../data/" + "example.csv"
    data = pd.read_csv(filename, usecols=[1, 4, 5, 6, 9, 12, 13, 14])
    data["IV"] = data["IV"].str.rstrip('%').astype(
        'float') / 100.0   # transforms the percentage into decimal
    data["Spread"] = (data.Ask - data.Bid)   # spread column

    CALL = data[data.Type == "Call"].reset_index(drop=True)
    PUT = data[data.Type == "Put"].reset_index(drop=True)

    # 取市场数据
    S0 = 312.23  # stockname的当前股价
    T = data.DTE[0]/365  # 注意daycount的问题
    r = 0.015  # 应取相应期限的risk free rate
    diff = 0.015  # drift，r-q-repo

    # 计算implied vol
    IV_call = CALL.parallel_apply(lambda x: implied_vol_minimize(
        x['Midpoint'], S0, x['Strike'], T, r, diff, payoff="call", disp=False), axis=1)
    CALL = CALL.assign(IV_mid=IV_call.values)
    IV_put = PUT.parallel_apply(lambda x: implied_vol_minimize(
        x['Midpoint'], S0, x['Strike'], T, r, diff, payoff="put", disp=False), axis=1)
    PUT = PUT.assign(IV_mid=IV_put.values)
    CALL = CALL[CALL.IV_mid != -1].reset_index(drop=True)
    PUT = PUT[PUT.IV_mid != -1].reset_index(drop=True)

    if ExpValue > S0:
        print("Implied vs Provided: ", np.linalg.norm(
            CALL['IV_mid']-CALL['IV'], 1))  # 比较拿到的vol数据与计算的vol的差异
        Price_call = CALL.parallel_apply(lambda x: LSMWO(
            S0, x['Strike'], T, r, diff, x['IV_mid'], "call", ExpValue, ExpTime), axis=1)
        CALL = CALL.assign(Exp_Price=Price_call.values)
        CALL['ratio'] = CALL['Exp_Price']/CALL['Ask']
        Choice = CALL[CALL.ratio == CALL.ratio.max()]
    else:
        print("Implied vs Provided: ", np.linalg.norm(
            PUT['IV_mid']-PUT['IV'], 1))
        Price_put = PUT.parallel_apply(lambda x: LSMWO(
            S0, x['Strike'], T, r, diff, x['IV_mid'], "put", ExpValue, ExpTime), axis=1)
        PUT = PUT.assign(Exp_Price=Price_put.values)
        PUT['ratio'] = PUT['Exp_Price']/PUT['Ask']
        Choice = PUT[PUT.ratio == PUT.ratio.max()]
    return Choice


if __name__ == "__main__":
    # import os
    # os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

    # import cProfile, pstats, io

    # pr = cProfile.Profile()
    # pr.enable()

    Choice_opinion("STOCK.O", 280, 30)

    # pr.disable()
    # s = io.StringIO()
    # sortby = "cumtime"  # 仅适用于 3.6, 3.7 把这里改成常量了
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())
    # pr.dump_stats("pipeline2.prof")
