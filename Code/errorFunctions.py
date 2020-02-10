# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020

from matplotlib import pyplot as plt
from arch.bootstrap import MCS
from scipy.stats import ttest_ind
import numpy as np
from decimal import Decimal


def defineListyle(evaluation):

    lstmColors = ["crimson", "orangered", "salmon", "darkred"]

    esnColors = [
        "blue",
        "navy",
        "darkblue",
        "royalblue",
        "indigo",
        "deeskyblue",
        "cornflowerblue",
    ]

    if evaluation.modelType == "HAR":
        color = "dimgray"
        linestyle = "-"
        marker = "None"
    if evaluation.modelType in ["ESN", "EchoStateExperts"]:
        color = esnColors.pop(0)
        linestyle = ":"
        marker = "*"
        if evaluation.modelType == "EchoStateExperts":
            linestyle = "-"
            color = "midnightblue"
    if evaluation.modelType in ["LSTM", "LSTMExperts"]:
        color = lstmColors.pop(0)
        linestyle = "-."
        marker = "*"
        if evaluation.modelType == "LSTMExperts":
            linestyle = "-"
            color = "red"
    if evaluation.modelType in ["HybridExperts"]:
        color = "green"
        linestyle = "-"
        marker = "*"

    return color, linestyle, marker


def plotErrorVectors(
    evalResults, errorType, testingRangeStartDate, testingRangeEndDate, assetList, data
):
    alpha = 0.25
    plt.rcParams["font.family"] = "Times New Roman"

    # assert evalResults[0].modelName == "HAR"
    benchmark = evalResults[0]

    fig = plt.figure()
    chart = fig.add_subplot(111)
    chart.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    errorMatricesList = [
        np.array(evaluation.errorMatrix[errorType]).flatten().reshape(-1, 1)
        for evaluation in evalResults
    ]

    try:
        mcs = MCS(np.concatenate(errorMatricesList, axis=1), size=0.1)
        mcs.compute()
        mcsAvailable = True
    except:
        mcsAvailable = False
        mcsLabel = ""
        pValue = "n.a."

    for i in range(len(evalResults)):

        evaluation = evalResults[i]
        if mcsAvailable:
            pValue = mcs.pvalues["Pvalue"][i]

        # T-Test
        tTestResult = ttest_ind(
            evaluation.errorMatrix[errorType], benchmark.errorMatrix[errorType]
        )

        # Colors, labeling and line styles
        significantPoints = list(np.where(tTestResult[1] < alpha)[0])
        label = evaluation.modelName + " p: " + str(pValue)
        color, linestyle, marker = defineListyle(evaluation)

        chart.plot(
            evaluation.errorVector[errorType],
            label=label,
            marker=marker,
            color=color,
            linestyle=linestyle,
            linewidth=1,
            markevery=significantPoints,
        )
    chartTitle = " 30 Days Ahead Forecasting Error \n %d/%d - %d/%d  \n Assets: " % (
        testingRangeStartDate.month,
        testingRangeStartDate.year,
        testingRangeEndDate.month,
        testingRangeEndDate.year,
    )

    chart.set_title(
        errorType
        + chartTitle
        + str(assetList).replace("[", "").replace("]", "").replace("'", "")
    )

    plt.legend(loc="upper left")
    plt.xlabel("Days Ahead")
    plt.tight_layout()
    plt.show()

    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    chartTitleOneDayError = (
        " 1 Day Ahead "
        + errorType
        + " Forecasting Error \n %d/%d - %d/%d  \n Assets: "
        % (
            testingRangeStartDate.month,
            testingRangeStartDate.year,
            testingRangeEndDate.month,
            testingRangeEndDate.year,
        )
    )

    for evaluation in evalResults:
        color, linestyle, marker = defineListyle(evaluation)
        oneDayAheadErrorVector = evaluation.oneDayAheadError[errorType]
        plt.plot(
            oneDayAheadErrorVector,
            label=evaluation.modelName
            + " \u03BC: %.2e \u03C3: %.2e"
            % (np.average(oneDayAheadErrorVector), np.std(oneDayAheadErrorVector)),
            color=color,
            linestyle=linestyle,
            linewidth=1,
        )
    plt.title(
        chartTitleOneDayError
        + str(assetList).replace("[", "").replace("]", "").replace("'", "")
    )
    plt.xlabel("Days")
    plt.legend(loc="upper left")
    plt.show()
