import os
from re import A

import numpy as np
import pandas as pd
import scipy.stats
from factor_covariance_adjustment import FactorCovAdjuster
from matplotlib import pyplot as plt
from utils import draw_eigvals_edf, num_eigvals_explain

if __name__ == "__main__":
    dir_main = r"E:\Others\Programming\Python\python_work\CaishengTech" + "/"
    dir_data = dir_main + "data/"
    dir_result = dir_main + "result/"
    dir_analysis = dir_main + "analysis/"

    # Analysis
    df_eigvals = pd.read_csv(dir_analysis + "df_eigvals.csv", index_col=0)
    # df_eigvals = df_eigvals.div(df_eigvals.sum(axis=1), axis=0) * 100

    df_return_day = pd.read_csv(dir_analysis + "df_return_day.csv", index_col=0)
    idx = list(set(df_eigvals.index) & set(df_return_day.index))
    df_eigvals = df_eigvals.loc[idx]
    df_return_day = df_return_day.loc[idx]

    vols_ = df_return_day.std(axis=1).values

    nums = []
    for i in df_eigvals.index:
        nums.append(num_eigvals_explain(0.8, df_eigvals.loc[i].values))
        # a = df_eigvals.loc[i].values
        # s = 0
        # for j in a:
        #     if j > 1.2:
        #         s += 1
        # nums.append(s)
    nums_ = np.array(nums)
    print(nums_)

    for new_periods in range(1, 31):
        vols, nums = vols_.copy(), nums_.copy()
        vols_new, nums_new = [], []
        sv, sn = 0, 0
        for i in range(642 - new_periods):
            sv += vols[i]
            sn += nums[i]
            if i % new_periods == new_periods - 1:
                vols_new.append(sv)
                nums_new.append(sn)
                sv, sn = 0, 0
        vols = np.array(vols_new) / new_periods
        nums = np.array(nums_new) / new_periods

        coef = np.corrcoef(vols, nums)[0, 1]
        print(new_periods, coef)
        nums = np.sign(nums[1:] - nums[:-1])
        vols = np.sign(vols[1:] - vols[:-1])
        correct = (nums / vols + 1) / 2
        correct[correct == 0.5] = 0
        # print(correct.sum() / len(correct))

    # Draw
    ax1 = plt.gca()
    ax1.plot(vols, color="r", label="rets")
    ax2 = ax1.twinx()
    ax2.plot(nums, color="b", label="vols")
    plt.show()
