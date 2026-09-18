# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for plotting portfolio component gross returns."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import zeros

generatePlots = True
x = np.nan
dataset_name = "sp500"
data_path = f"../data/stock_data/{dataset_name}.csv"


def sign(x):
    return int(x > 0) - int(x < 0)


def getHistPrices(lab, w, length, start="2022-01-01", end="2024-01-01"):
    df = pd.read_csv(data_path, index_col=0)
    df.shape
    df = df.dropna(axis=1)
    df.shape
    # print(lab,w,start,end)
    df = df[list(lab)].loc[start:end]
    return df.to_numpy(), df.index


def plotMultSeriesLgSh(
    prices,
    lab,
    w,
    dates,
    isShow=False,
    isPlotMean=False,
    isShowPortv=False,
    portv=np.nan,
    linewidth=1,
    main="",
):
    # Use a single plot of gross returns beginning at y=1 for all securities in lab
    plt.rcParams["figure.figsize"] = [12, 8]
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots()
    for i in range(prices.shape[1]):
        if w[i] > 0:
            grossretvec = prices[:, i] / prices[0, i]
        elif w[i] < 0:
            grossretvec = prices[0, i] / prices[:, i]
        ax.plot(dates, grossretvec, linewidth=linewidth, label=lab[i])
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(main)
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    # print(lab[i],prices[prices.shape[0]-1, i]/prices[0, i]) #linewidth=3, color='lime')
    plt.title(main)
    if (
        isPlotMean
    ):  # this assumes equal weighting because otherwise we have weightPortOOS() below.
        meanVec = (prices[:, :] / prices[0, :]).mean(axis=1)
        plt.plot(meanVec, color="black", linewidth=3)  # can be limegreen
    if isShowPortv:
        print(dates.shape, portv.shape)
        ax.plot(dates, portv, color="black", linewidth=2.5)
    if isShow:
        plt.show()


def weightPortOOS(
    lab,
    length,
    D,
    w,
    prices=np.nan,
    dates=np.nan,
    start="2025-11-29",
    end="2025-11-28",
    isNaive=False,
    cached=np.nan,
):  # len x D prices
    if np.isnan(prices).any():  # then nan
        print(lab)
        prices, dates = getHistPrices(lab, w, length, start=start, end=end)
    numNonZeroWs = sum(w)
    portv = zeros([length])
    for i in range(length):
        for d in range(D):  # roll down a return line
            if isNaive:
                portv[i] = (
                    portv[i] + (1 / numNonZeroWs) * prices[i, d] / prices[0, d]
                )  # ignore w
            elif w[d] > 0:
                portv[i] = portv[i] + w[d] * prices[i, d] / prices[0, d]
            elif w[d] < 0:  # allow neg weights w[i]
                portv[i] = portv[i] + w[d] * prices[i, d] / prices[0, d]
    return (portv), dates


if True:
    lab = ("FICO", "LLY", "CAH", "OXY", "STLD")
    w = np.tile(1 / len(lab), len(lab))


def displayComponents(
    lab, w, length, start, end, isShow=True, isShowPortv=False, portv=np.nan
):
    mainstr = (
        "In-Sample Portfolio Components as Gross Returns for "
        + str(len(w))
        + " Stocks with Weights"
    )
    prices, dates = getHistPrices(lab, w, length, start, end)
    # print("prices.shape",prices.shape)
    portv, dates = weightPortOOS(
        lab, length, D=len(w), w=w, dates=dates, prices=prices, start=start, end=end
    )
    plotMultSeriesLgSh(
        prices,
        lab,
        w,
        dates,
        isShow=isShow,
        isPlotMean=False,
        isShowPortv=isShowPortv,
        portv=portv,
        main=mainstr,
    )
    return portv
