# Code Appendix
# Masterthesis: Forecasting Realized Covariance with LSTM and Echo State Networks
# Author: Lukas Schreiner, 2020

from matplotlib import pyplot as plt
from arch.bootstrap import MCS
from scipy.stats import ttest_ind
import numpy as np

root_path = "./gdrive/My Drive/Colab Notebooks/Data/"


def generator(input_list):
    while True:
        for color in input_list:
            yield color


def defineListyle(evaluation, lstmColors, esnColors):

    if evaluation.modelType == "HAR":
        color = "dimgray"
        linestyle = "-"
        marker = "None"
    if evaluation.modelType in ["ESN", "EchoStateExperts"]:
        color = esnColors.__next__()
        linestyle = ":"
        marker = "*"
        if evaluation.modelType == "EchoStateExperts":
            linestyle = "-"
            color = "midnightblue"
    if evaluation.modelType in ["LSTM", "LSTMExperts"]:
        color = lstmColors.__next__()
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


lstmColors = generator(["crimson", "orangered", "salmon", "darkred"])
esnColors = generator(
    ["cornflowerblue", "blue", "navy", "darkblue", "royalblue", "indigo"]
)


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

        color, linestyle, marker = defineListyle(evaluation, lstmColors, esnColors)

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

    chartTitle_30d = (
        errorType
        + chartTitle
        + str(assetList).replace("[", "").replace("]", "").replace("'", "")
    )
    chart.set_title(chartTitle_30d)

    plt.legend(loc="upper left")
    plt.xlabel("Days Ahead")
    plt.tight_layout()
    plt.savefig(errorType + "30d" + ".png", dpi=400)
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
        color, linestyle, marker = defineListyle(evaluation, lstmColors, esnColors)
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
    chart_title_1d = chartTitleOneDayError + str(assetList).replace("[", "").replace(
        "]", ""
    ).replace("'", "")
    plt.title(chart_title_1d)
    plt.xlabel("Days")
    plt.legend(loc="upper left")
    plt.savefig(errorType + "1d" + ".png", dpi=400)
    plt.show()
