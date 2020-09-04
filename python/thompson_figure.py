#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figures for the Thompson Sampling blog post here:

    https://gertjanvandenburg.com/blog/thompson_sampling/

The output JSON file is intended to be used with the LineGraph.js code here:

    https://github.com/GjjvdBurg/LineGraph.js

Author: Gertjan van den Burg
License: See LICENSE file
Copyright: 2020, G.J.J. van den Burg

"""

import argparse
import json
import matplotlib.pyplot as plt

from k_armed_testbed import (
    TestBed,
    ThompsonSampling,
    UpperConfidence,
    run_experiment,
    make_regret_plot,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Output JSON file to write to")
    parser.add_argument(
        "--show-plot", help="Show plot before saving", action="store_true"
    )
    parser.add_argument(
        "plot",
        help="Plot to make",
        choices=["hyperparam", "compare_ucb"],
        required=True,
    )
    return parser.parse_args()


def hyperparameters(
    k=10, repeats=1000, steps=10000, cache_dir=None, show_plot=True
):
    env = TestBed(k=k, baseline=0, sigma=None)

    bandits = []

    hyperparam = [
        (0, 1, 0.02, 0.02),
        (0, 1, 1, 1),  # "just-put-one" prior
        (10, 1, 1, 1),  # wrong mean
        (0, 10, 1, 1),  # overestimate variance
        (0, 0.1, 1, 1),  # underestimate variance
        (0, 1, 0.02, 1),  # low alpha
        (0, 1, 100, 1),  # high alpha
        (0, 1, 100, 100),  # high alpha and beta
    ]

    for m, v, alpha, beta in hyperparam:
        ts = ThompsonSampling(k=k, m_a=m, v_a=v, alpha_a=alpha, beta_a=beta)
        bandits.append(ts)

    _, _, avg_regret = run_experiment(
        env, bandits, repeats, steps, cache_dir=cache_dir, pseudo_regret=True,
    )

    if show_plot:
        fig, axes = plt.subplots(1, 1)
        make_regret_plot(axes, avg_regret, bandits)
        plt.show()

    labels = [b.label() for b in bandits]

    return avg_regret, labels


def compare_ucb(
    k=10, repeats=1000, steps=10000, cache_dir=None, show_plot=True
):
    env = TestBed(k=k, baseline=0, sigma=None)
    bandits = [
        ThompsonSampling(m_a=0, v_a=1, alpha_a=1, beta_a=1),
        ThompsonSampling(m_a=0, v_a=1, alpha_a=0.02, beta_a=0.02),
        ThompsonSampling(m_a=0, v_a=1, alpha_a=100, beta_a=100),
        UpperConfidence(c=1),
        UpperConfidence(c=2),
        UpperConfidence(c=0.5),
    ]

    _, _, avg_regret = run_experiment(
        env, bandits, repeats, steps, cache_dir=cache_dir, pseudo_regret=True,
    )

    if show_plot:
        fig, axes = plt.subplots(3, 1)
        make_regret_plot(axes[2], avg_regret, bandits)
        plt.show()

    labels = [b.label() for b in bandits]
    return avg_regret, labels


def main():
    args = parse_args()

    if args.plot == "hyperparam":
        regret, labels = hyperparameters(
            repeats=1000,
            steps=10001,
            cache_dir=args.cache_dir,
            show_plot=args.show_plot,
        )
    elif args.plot == "compare_ucb":
        regret, labels = compare_ucb(
            repeats=2000,
            steps=10001,
            cache_dir=args.cache_dir,
            show_plot=args.show_plot,
        )

    out = {"meta": {"xlabel": "Time", "ylabel": "Regret",}, "data": {}}
    out["meta"]["ymax"] = 1000

    regret = regret.T
    data = {}
    data["X"] = list(range(1, 1 + regret.shape[0]))
    series = []
    for l, label in enumerate(labels):
        series.append({"name": label, "values": list(regret[:, l])})
    data["series"] = series
    out["data"] = data

    with open(args.export, "w") as fp:
        json.dump(out, fp)


if __name__ == "__main__":
    main()
