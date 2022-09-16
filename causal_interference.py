#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import operator
from functools import reduce


def causal_search(df, features, N):
    """
    quantifies how much impact the intervention from HR has on the different
    groups of employees, defined by their features

    :param df: pandas dataframe object containing the data
    :param features: list of strings of features that are included in the search
    :return: list of tuples (list of features, average decline of number of workdays,
             size subgroup/total size), sorted by the average decline of number of workdays
    """
    result = []
    for combo_features in itertools.combinations(features, N):
        target = [list(df[feature].unique()) for feature in combo_features]
        for subgroup in itertools.product(*target):
            filt_list = [(df[combo_features[i]] == subgroup[i]) for i in range(N)]
            filt = reduce(lambda x, y: operator.and_(x, y), filt_list)
            subgroup_share = len(df[filt]) / len(df)
            result.append(
                (
                    subgroup,
                    df[filt]["wordkays/week (before)"].mean()
                    - df[filt]["wordkays/week (after)"].mean(),
                    subgroup_share,
                )
            )
    return sorted(result, key=lambda x: x[1], reverse=True)


def text_output(result_green, result_red, k):
    print(
        "The members from team green who lost the most motivation after the change are: "
    )
    for i in range(k):
        print(
            "{}: the group {} with value {} and distribution {}, ".format(
                i + 1,
                result_green[i][0],
                result_green[i][1],
                result_green[i][2],
            )
        )
    print("-------------------------------")
    print(
        "The members from team green who lost the least motivation after the change are: "
    )
    for i in range(k):
        print(
            "{}: the group {} with value {} and distribution {}, ".format(
                i + 1,
                result_green[-i - 1][0],
                result_green[-i - 1][1],
                result_green[-i - 1][2],
            )
        )
    print("-------------------------------")
    print(
        "The members from team red who gained the most motivation after the change are: "
    )
    for i in range(k):
        print(
            "{}: the group {} with value {} and distribution {}, ".format(
                i + 1,
                result_red[-i - 1][0],
                -result_red[-i - 1][1],
                result_red[-i - 1][2],
            )
        )
    print("-------------------------------")
    print(
        "The members from team red who gained the least motivation after the change are: "
    )
    for i in range(k):
        print(
            "{}: the group {} with value {} and distribution {}, ".format(
                i + 1,
                result_red[i][0],
                -result_red[i][1],
                result_red[i][2],
            )
        )


def visualization(result_green, result_red, k):

    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    subgroups_green = tuple(result_green[i][0][0] for i in range(k)) + tuple(
        result_green[-i - 1][0][0] for i in reversed(range(k))
    )
    y_pos = np.arange(2 * k)
    value_green = [result_green[i][1] for i in range(k)] + [
        result_green[-i - 1][1] for i in reversed(range(k))
    ]

    ax.barh(y_pos, value_green, align="center", color="green")
    ax.set_yticks(y_pos, labels=subgroups_green)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Difference Average # Working Days")
    ax.set_title("Effect of HR Intervention on Green Team")

    plt.savefig("green_team_1.pdf")

    plt.rcdefaults()
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    subgroups_red = tuple(result_red[-i - 1][0][0] for i in range(k)) + tuple(
        result_red[i][0][0] for i in reversed(range(k))
    )
    value_red = [-result_red[-i - 1][1] for i in range(k)] + [
        -result_red[i][1] for i in reversed(range(k))
    ]

    ax.barh(y_pos, value_red, align="center", color="red")
    ax.set_yticks(y_pos, labels=subgroups_red)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel("Difference Average # Working Days")
    ax.set_title("Effect of HR Intervention on Red Team")

    plt.savefig("red_team_1.pdf")


def main():
    data = "workingdays.csv"
    df = pd.read_csv(data)

    # check effect that intervention had on team red and team green
    result = (
        df[["shift", "wordkays/week (before)", "wordkays/week (after)"]]
        .groupby("shift")
        .mean()
    )
    print(result)

    # preprocessing data
    # putting numerical features into equal sized bins
    df["CommuteRange"] = pd.qcut(
        df["commute distance (km)"],
        [0, 0.25, 0.5, 0.75, 1.0],
        ["1-5", "5-10", "10-15", "15-19"],
    )
    # df["AgeRange"] = pd.qcut(
    #    df["age"], [0, 0.25, 0.5, 0.75, 1.0], ["23-29", "29-33", "33-47", "47-60"]
    # )
    df["AgeRange"] = pd.cut(df["age"], [0, 31, np.inf], labels=["<32", ">=32"])

    # define the features that we want to investigate
    ft = ["CommuteRange", "experience", "education", "nationality", "AgeRange"]
    # ft = ["CommuteRange", "education", "nationality", "AgeRange"]

    # set the number of features that each subgroup will be defined by
    N = 1

    # first, find the members of team green that lost the most and least motivation
    # then, find the members of team red that gained the most and least motivation
    result_green = causal_search(df[df["shift"] == "green"], ft, N)
    result_red = causal_search(df[df["shift"] == "red"], ft, N)

    # text output
    text_output(result_green, result_red, 10)

    # visualized output
    visualization(result_green, result_red, 5)


if __name__ == "__main__":
    main()
