#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Little test to explore the intuition behind UCB.

Plot how the estimated action value evolves over time.

We find that the time between draws of the suboptimal arms becomes larger and 
larger (depending on c and noise variance), while the upper bound estimate of 
the best arm becomes increasingly accurate.

Author: Gertjan van den Burg
License: See LICENSE file
Copyright: 2020, G.J.J. van den Burg

"""

import argparse
import json
import tqdm

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="JSON file to write to")
    parser.add_argument(
        "-s", "--seed", help="Random seed", default=42, type=int
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    T = 1000
    c = 1.0
    n_arms = 3
    noisevar = 0.1

    Nt = {k: 0 for k in range(1, n_arms + 1)}
    means = {k: float(k) for k in range(1, n_arms + 1)}
    obs = {k: [] for k in range(1, n_arms + 1)}

    Av = np.zeros((T, n_arms))

    # run each arm at least once
    for a in Nt:
        r = means[a] + np.random.normal(0, noisevar)
        obs[a].append(r)
        Av[0, a - 1] = r
        Nt[a] += 1

    for t in tqdm.trange(2, T + 1):
        for i, a in enumerate(means):
            # noisy reward
            r = means[a] + np.random.normal(0, noisevar)
            obs[a].append(r)
            estimate = np.mean(obs[a]) + c * np.sqrt(np.log(t) / Nt[a])
            Av[t - 1, i] = estimate

        amax = np.argmax(Av[t - 1, :]) + 1
        Nt[amax] += 1

    print(f"Number of draws per arm: {Nt}")

    if args.output:
        out = {"meta": {"xlabel": "Time", "ylabel": "Estimated action value"}}
        data = {"X": list(range(1, T + 1))}
        series = []
        for i, a in enumerate(means):
            series.append({"name": f"μ = {a}", "values": list(Av[:, i])})
        data["series"] = series
        out["data"] = data
        with open(args.output, "w") as fp:
            json.dump(out, fp)
    else:
        for i, a in enumerate(means):
            plt.plot(Av[:, i], label=f"$\mu = {a}$")
        plt.ylabel("Estimated action value")
        plt.xlabel("Time")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
