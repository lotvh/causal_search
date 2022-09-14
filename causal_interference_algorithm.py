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
            subgroup_share = len(df[filt])/len(df)
            result.append(
                (
                    subgroup,
                    df[filt]["wordkays/week (before)"].mean()
                    - df[filt]["wordkays/week (after)"].mean(),
                    subgroup_share
                )
            )
    return sorted(result, key=lambda x: x[1], reverse=True)

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
    commute_bins = [0, 5, 10, 15, 20]
    commute_groups = ["1-5", "6-10", "11-15", "16-20"]
    age_bins = [20, 30, 40, 50, 60]
    age_groups = ["21-30", "31-40", "41-50", "51-60"]
    df["CommuteRange"] = pd.cut(
        df["commute distance (km)"], commute_bins, labels=commute_groups
    )
    df["AgeRange"] = pd.cut(df["age"], age_bins, labels=age_groups)

    # define the features that we want to investigate
    ft = ["CommuteRange", "experience", "education", "nationality", "AgeRange"]

    # set the number of features that each subgroup will be defined by
    N = 1

    # first, find the members of team green that lost the most and least motivation
    k = 5
    result_green = causal_search(df[df["shift"] == "green"], ft, N)
    print(
        "The members from team green who lost the most motivation after the change are: {}".format(
            result_green[:k]
        )
    )
    print(
        "The members from team green who lost the least motivation after the change are: {}".format(
            result_green[-k:]
        )
    )

    # then, find the members of team red that gained the most and least motivation
    result_red = causal_search(df[df["shift"] == "red"], ft, N)
    print(
        "The members from team red who gained the most motivation after the change are: {}".format(
            result_red[-k:]
        )
    )
    print(
        "The members from team red who gained the least motivation after the change are: {}".format(
            result_red[:k]
        )
    )


if __name__ == "__main__":
    main()
