import os

import numpy as np
import pandas as pd
import scipy.stats
from factor_covariance_adjustment import FactorCovAdjuster
from matplotlib import pyplot as plt
from utils import draw_eigvals_edf, num_eigvals_explain

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)


def find_valid_stocks(dir_input: str) -> set[str]:
    ttl_code = set()
    for root, dirs, files in os.walk(dir_input):
        tmp_code = set()
        for file in files:
            if root.endswith("SH"):
                tmp_code.add(file[:6])
        if tmp_code:
            if root.endswith("20190430\SH"):
                ttl_code = tmp_code
            else:
                ttl_code = ttl_code & tmp_code
    return sorted(list(ttl_code))


def get_returns_min(dir_input: str, dir_output: str, valid_stocks: list[str]) -> None:
    for root, dirs, files in os.walk(dir_input):
        df_return = pd.DataFrame(columns=list(map(str, range(237))))
        try:
            for file in files:
                if root.endswith("SH"):
                    code, date = file[:6], root[-11:-3]
                    if code in valid_stocks:
                        df = pd.read_parquet(root + "/" + file)
                        # 9:31 - 14:56: 237
                        returns = (df["close"] / df["close"].shift() - 1)[1:238] * 100
                        df_return.loc[code] = returns.values
        except:
            continue
        if not df_return.empty:
            df_return = df_return.reindex(valid_stocks)  # in the same order
            df_return.to_parquet(dir_output + f"df_return_{date}.parquet")
            print(date)


def get_returns_day(dir_input: str, dir_output: str, valid_stocks: list[str]) -> None:
    df_return = pd.DataFrame(columns=["code", "date", "close"])
    for root, dirs, files in os.walk(dir_input):
        if root.endswith("SH"):
            date = root[-11:-3]
            print(date)
            for file in files:
                code = file[:6]
                if code in valid_stocks:
                    df = pd.read_parquet(root + "/" + file)
                    data = df.loc[len(df) - 1, ["code", "date", "close"]]
                    df_return.loc[len(df_return)] = data
    df_return.set_index(["code", "date"], inplace=True)
    df_return = df_return.unstack("code")
    df_return = (df_return / df_return.shift() - 1) * 100
    df_return.dropna(axis=0, inplace=True)
    df_return = df_return.T
    df_return.index = [str(i[1]) for i in df_return.index]
    df_return.columns = [str(j) for j in df_return.columns]
    df_return = df_return.reindex(valid_stocks)
    df_return = df_return.T
    df_return.to_csv(dir_output + "df_return_day.csv")


def analysis(dir_input: str, dir_output: str, stock_num: int, params: dict[str:dict]):
    for root, dirs, files in os.walk(dir_input):
        df_eigvals = pd.DataFrame(columns=list(map(str, range(stock_num))))
        for i, file in enumerate(files):
            print(i)
            print(file)
            FRM = pd.read_parquet(root + "/" + file).values
            adjuster = FactorCovAdjuster(FRM)
            F_Raw = adjuster.calc_fcm_raw(**params["cfr"])
            F_NW = adjuster.newey_west_adjust(**params["nwa"])
            F_Eigen = adjuster.eigenfactor_risk_adjust(**params["era"])
            prev_FCM = adjuster.FCM / params["nwa"]["multiplier"]
            if i == 0:
                continue
            F_VRA = adjuster.volatility_regime_adjust(prev_FCM, **params["vra"])
            try:
                eigvals = np.linalg.eigvalsh(adjuster.FCM)
            except:
                df_eigvals.to_csv(dir_output + "df_eigvals.csv")
            df_eigvals.loc[file[-16:-8], :] = eigvals
            # # fig
            # # draw_eigvals_edf(F_Raw, **params["fig"], label="F_Raw")
            # draw_eigvals_edf(F_NW, **params["fig"], label="F_NW")
            # draw_eigvals_edf(F_Eigen, **params["fig"], label="F_Eigen")
            # draw_eigvals_edf(F_VRA, **params["fig"], label="F_VRA")
            # plt.legend()
            # plt.show()
        if not df_eigvals.empty:
            df_eigvals.to_csv(dir_output + "df_eigvals.csv")


if __name__ == "__main__":
    dir_main = r"E:\Others\Programming\Python\python_work\CaishengTech" + "/"
    dir_data = dir_main + "data/"
    dir_result = dir_main + "result/"
    dir_analysis = dir_main + "analysis/"

    # # Preprocess
    # valid_stocks = find_valid_stocks(dir_data)
    # print("number of valid stocks:", len(valid_stocks))

    # # Get data
    # get_returns_min(dir_data, dir_result, valid_stocks)
    # get_returns_day(dir_data, dir_analysis, valid_stocks)

    # Set parameters
    stock_num = 137
    multiplier = 237  # number of periods in a FCM frequency
    params = {
        "cfr": {"half_life": multiplier * 21},
        "nwa": {"half_life": multiplier * 21, "max_lags": 2, "multiplier": multiplier},
        "era": {"coef": 1.4, "M": 100},
        "vra": {"half_life": multiplier * 5},
        "fig": {"x_range": np.linspace(0, 100, 500), "bandwidth": 1},
    }

    # Calculate eigenvalues
    # analysis(dir_result, dir_analysis, stock_num, params)

    # Analysis
    df_eigvals = pd.read_csv(dir_analysis + "df_eigvals.csv", index_col=0)
    # df_eigvals = df_eigvals.div(df_eigvals.sum(axis=1), axis=0) * 100

    df_return_day = pd.read_csv(dir_analysis + "df_return_day.csv", index_col=0)
    idx = list(set(df_eigvals.index) & set(df_return_day.index))
    df_eigvals = df_eigvals.loc[idx]
    df_return_day = df_return_day.loc[idx]

    length = 1000
    eig_order = 0
    pct = 0.85
    new_periods = 4

    for periods in range(1, 6):
        new_periods = periods

        rets = df_return_day.mean(axis=1).values[new_periods:][:length]
        vols = df_eigvals[f"{eig_order}"].values[:-new_periods][:length]
        nums = []
        for i in df_eigvals.index:
            nums.append(num_eigvals_explain(pct, df_eigvals.loc[i].values))
        nums = np.array(nums)[:-1]

        rets_new, vols_new, nums_new = [], [], []
        sr, sv, sn = 0, 0, 0
        for i in range(len(rets)):
            sr += rets[i]
            sv += vols[i]
            sn += nums[i]
            if i % new_periods == new_periods - 1:
                rets_new.append(sr)
                vols_new.append(sv)
                nums_new.append(sn)
                sr, sv, sn = 0, 0, 0
        rets = np.array(rets_new) / new_periods
        vols = np.array(vols_new) / new_periods
        nums = np.array(nums_new) / new_periods

        corr_ret_vol = scipy.stats.pearsonr(rets, vols)[0]
        corr_ret_num = scipy.stats.pearsonr(abs(rets), nums)[0]
        # print("corr_ret_vol", f"({str(new_periods)}) ".ljust(5), round(corr_ret_vol, 3))
        print("corr_ret_num", f"({str(new_periods)}) ".ljust(5), round(corr_ret_num, 3))

    # Draw
    ax1 = plt.gca()
    ax1.plot(abs(rets), color="r", label="rets")
    ax2 = ax1.twinx()
    ax2.plot(nums, color="b", label="vols")
    plt.show()
