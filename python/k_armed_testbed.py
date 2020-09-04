#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bandit Algorithms

This script follows Chapter 2 of Sutton and Barto (2nd) and simply reproduces 
figures 2.2 to 2.5.

Author: Gertjan van den Burg
License: MIT
Copyright: (c) 2020, G.J.J. van den Burg

"""

import abc
import hashlib
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tqdm

from matplotlib import ticker
from scipy.special import logsumexp


class TestBed:
    """ k-Armed Test Bed """

    def __init__(self, k=10, baseline=0, sigma=1.0):
        self.k = k
        self.baseline = baseline
        self.sigma = sigma
        self._opt_action = None

    @property
    def opt_action(self):
        if self._opt_action is None:
            raise ValueError("Not initialised properly!")
        return self._opt_action

    def step(self, action):
        mean = self._qstar[action]
        std = self._stdevs[action]
        return random.gauss(mean, std)

    def reset(self):
        self._qstar = []
        self._stdevs = []
        for _ in range(self.k):
            self._qstar.append(random.gauss(self.baseline, 1))
            if self.sigma is None:
                self._stdevs.append(random.uniform(0.5, 4.0))
            else:
                self._stdevs.append(self.sigma)
        self._opt_action = argmax(lambda a: self._qstar[a], range(self.k))
        return self

    def _key(self):
        h = hashlib.blake2b(digest_size=10)
        h.update(self.serialize())
        return h.hexdigest()

    def serialize(self):
        d = {"baseline": self.baseline, "_qstar": self._qstar}
        return json.dumps(d)

    @classmethod
    def deserialize(cls, s):
        d = json.loads(s)
        obj = cls(k=len(d["_qstar"]), baseline=d["baseline"])
        obj._qstar = d["_qstar"]
        return obj


class Bandit(metaclass=abc.ABCMeta):
    def __init__(self, k=10, initial_value=0, stepsize="avg"):
        self.k = k
        self.initial_value = initial_value
        self.stepsize = stepsize

    def reset(self):
        # Reset the state of the bandit.
        self.Q = {a: self.initial_value for a in range(self.k)}
        self.N = {a: 0 for a in range(self.k)}
        if self.stepsize == "avg":
            self.alpha = lambda a: 1 if self.N[a] == 0 else 1.0 / self.N[a]
        else:
            self.alpha = lambda a: self.stepsize

    @abc.abstractmethod
    def get_action(self):
        """ Choose an action to take """

    def record(self, action, reward):
        """ Record the reward of the action taken """
        # Follows algorithm on page 32
        A, R = action, reward
        self.N[A] += 1
        self.Q[A] += self.alpha(A) * (R - self.Q[A])


class EpsilonGreedy(Bandit):
    def __init__(self, k=10, epsilon=0.1, initial_value=0, stepsize="avg"):
        super().__init__(k=k, initial_value=initial_value, stepsize=stepsize)
        self.epsilon = epsilon

    def get_action(self):
        if random.random() <= self.epsilon:
            return random.randint(0, self.k - 1)
        return argmax(lambda a: self.Q[a], range(self.k))

    def label(self):
        return r"ε-greedy (ε = %g, Q_1 = %g, α = %s)" % (
            self.epsilon,
            self.initial_value,
            self.stepsize,
        )


class UpperConfidence(Bandit):
    def __init__(self, k=10, c=2.0):
        super().__init__(k=k)
        self.c = c

    def reset(self):
        super().reset()
        self.t = 0

    def get_action(self):
        self.t += 1
        func = lambda a: self.Q[a] + self.c * math.sqrt(
            math.log(self.t) / self.N[a]
        )
        for a in range(self.k):
            # first pick all actions at least once
            if self.N[a] == 0:
                return a
        return argmax(func, range(self.k))

    def label(self):
        return r"UCB (c = %g)" % self.c

    def key(self):
        return f"UCB_c{self.c}.csv"


class GradientBandit(Bandit):
    def __init__(self, k=10, stepsize="avg", use_baseline=True):
        super().__init__(k=k, stepsize=stepsize)
        self.use_baseline = use_baseline

    def reset(self):
        super().reset()
        self.H = {a: 0 for a in range(self.k)}
        self.probs, self.Rtbar, self.t = None, 0, 0

    def get_action(self):
        self.t += 1
        lse = logsumexp(list(self.H.values()))
        self.probs = [math.exp(self.H[a] - lse) for a in range(self.k)]
        a = random.choices(list(range(self.k)), weights=self.probs, k=1)
        return a[0]

    def record(self, action, reward):
        At, Rt = action, reward
        for a in range(self.k):
            self.H[a] += (
                self.alpha(a) * (Rt - self.Rtbar) * ((At == a) - self.probs[a])
            )
        # Note that the choice of baseline is somewhat arbitrary, but the
        # average reward works well in practice. See discussion on page 40 of
        # Sutton & Barto.
        if self.use_baseline:
            self.Rtbar += 1 / self.t * (Rt - self.Rtbar)

    def label(self):
        bsln = "with" if self.use_baseline else "without"
        return r"Gradient (α = %s, %s baseline)" % (self.stepsize, bsln)


class ThompsonSampling(Bandit):
    def __init__(self, k=10, m_a=0, v_a=0, alpha_a=1, beta_a=1):
        super().__init__(k=k)
        self.m_a = m_a
        self.v_a = v_a
        self.alpha_a = alpha_a
        self.beta_a = beta_a

    def reset(self):
        self.N = {a: 0 for a in range(self.k)}
        self.mean = {a: 0 for a in range(self.k)}
        self.rho = {a: self.m_a for a in range(self.k)}
        self.ssd = {a: 0 for a in range(self.k)}
        self.beta_t_a = {a: self.beta_a for a in range(self.k)}

    def _draw_ig(self, alpha, beta):
        # draw from an inverse gamma with parameters alpha and beta
        try:
            return 1.0 / random.gammavariate(alpha, 1.0 / beta)
        except ZeroDivisionError:
            print("Failed for: " + self.label())
            raise

    def _draw_normal(self, mu, sigma2):
        # draw from a normal distribution with mean mu and *variance* sigma2
        return random.gauss(mu, math.sqrt(sigma2))

    def get_action(self):
        mus = []
        for a in range(self.k):
            sigma2_a = self._draw_ig(
                0.5 * self.N[a] + self.alpha_a, self.beta_t_a[a]
            )
            mu_a = self._draw_normal(
                self.rho[a], sigma2_a / (self.N[a] + self.v_a)
            )
            mus.append(mu_a)
        return argmax(lambda a: mus[a], range(self.k))

    def record(self, action, reward):
        At, Rt = action, reward
        old_N, old_mean = self.N[At], self.mean[At]
        self.N[At] += 1
        self.mean[At] += 1 / self.N[At] * (Rt - self.mean[At])
        self.rho[At] = (self.v_a * self.m_a + self.N[At] * self.mean[At]) / (
            self.v_a + self.N[At]
        )
        self.ssd[At] += (
            Rt ** 2 + old_N * old_mean ** 2 - self.N[At] * self.mean[At] ** 2
        )
        self.beta_t_a[At] = (
            self.beta_a
            + 0.5 * self.ssd[At]
            + (
                self.N[At]
                * self.v_a
                * (self.mean[At] - self.m_a) ** 2
                / (2 * (self.N[At] + self.v_a))
            )
        )

    def label(self):
        params = f"m = {self.m_a}, ν = {self.v_a}, α = {self.alpha_a}, β = {self.beta_a}"
        return "Thompson (%s)" % params

    def key(self):
        return f"TS_m{self.m_a}_nu{self.v_a}_alpha{self.alpha_a}_beta{self.beta_a}.csv"


def argmax(func, args):
    """Simple argmax function """
    m, inc = -float("inf"), None
    for a in args:
        if (v := func(a)) > m:
            m, inc = v, a
    return inc


def plot_styles(num):
    line_styles = ["solid", "dashed", "dashdot", "dotted"]
    num_styles = len(line_styles)
    cm = plt.get_cmap("tab20")
    for i in range(num):
        color = cm(i // num_styles * float(num_styles) / num)
        style = line_styles[i % num_styles]
        yield color, style


def plot_common(axis, data, bandits):
    for i, (color, style) in enumerate(plot_styles(data.shape[0])):
        lines = axis.plot(data[i, :])
        lines[0].set_color(color)
        lines[0].set_linestyle(style)

    # axis.plot(data.T)
    box = axis.get_position()
    axis.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axis.legend(
        [b.label() for b in bandits],
        loc="upper left",
        bbox_to_anchor=(1.05, 1.0),
        borderaxespad=0,
        fontsize="small",
    )
    axis.set_xlabel("Steps")


def make_reward_plot(axis, avg_rewards, bandits):
    plot_common(axis, avg_rewards, bandits)
    axis.set_ylabel("Average\nreward", rotation="horizontal", ha="center")


def make_optact_plot(axis, avg_optact, bandits):
    plot_common(axis, avg_optact, bandits)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axis.set_ylabel("%\nOptimal\naction", rotation="horizontal", ha="center")


def make_regret_plot(axis, avg_regret, bandits):
    plot_common(axis, avg_regret, bandits)
    axis.set_ylabel("Average\nregret", rotation="horizontal", ha="center")


def cache(items, name, key, cache_dir):
    dest = os.path.join(cache_dir, name + "_" + key)
    with open(dest, "a") as fp:
        fp.write(",".join(map(str, items)))
        fp.write("\n")


def run_experiment(
    env, bandits, repeats, steps, cache_dir=None, pseudo_regret=False
):
    B = len(bandits)
    rewards = np.zeros((B, repeats, steps))
    optact = np.zeros((B, repeats, steps))
    regret = np.zeros((B, repeats, steps))
    for r in tqdm.trange(repeats):
        # reset the bandits and the environment
        [bandit.reset() for bandit in bandits]
        env.reset()

        mu_star = env._qstar[env.opt_action]

        for t in range(1, steps + 1):
            for b in range(B):
                bandit = bandits[b]
                action = bandit.get_action()
                reward = env.step(action)
                bandit.record(action, reward)
                rewards[b, r, t - 1] = reward
                optact[b, r, t - 1] = action == env.opt_action
                if pseudo_regret:
                    if t > 1:
                        regret[b, r, t - 1] = (
                            regret[b, r, t - 2] + mu_star - env._qstar[action]
                        )
                    else:
                        regret[b, r, t - 1] = mu_star - env._qstar[action]
                else:
                    regret[b, r, t - 1] = t * mu_star - rewards[b, r, :].sum()

        if cache_dir is None:
            continue

        for b in range(B):
            bandit = bandits[b]
            cache(regret[b, r, :], "regret", bandit.key(), cache_dir)
            cache(optact[b, r, :], "optact", bandit.key(), cache_dir)
            cache(rewards[b, r, :], "rewards", bandit.key(), cache_dir)

    avg_rewards = rewards.mean(axis=1)
    avg_optact = optact.mean(axis=1)
    avg_regret = regret.mean(axis=1)
    return avg_rewards, avg_optact, avg_regret


def figure_2_2(k=10, repeats=2000, steps=1000, epsilons=None):
    env = TestBed(k=k)
    epsilons = epsilons or [0.1, 0.01, 0]
    bandits = [EpsilonGreedy(k=k, epsilon=e) for e in epsilons]
    avg_rewards, avg_optact, _ = run_experiment(env, bandits, repeats, steps)

    fig, axes = plt.subplots(2, 1)
    make_reward_plot(axes[0], avg_rewards, bandits)
    make_optact_plot(axes[1], avg_optact, bandits)
    plt.show()


def figure_2_3(k=10, repeats=2000, steps=1000):
    env = TestBed(k=k)
    bandits = [
        EpsilonGreedy(k=k, epsilon=0.1, initial_value=0, stepsize=0.1),
        EpsilonGreedy(k=k, epsilon=0, initial_value=5, stepsize=0.1),
    ]
    _, avg_optact, _ = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_optact_plot(axis, avg_optact, bandits)
    plt.show()


def figure_2_4(k=10, repeats=2000, steps=1000, c=2):
    env = TestBed(k=k)
    bandits = [EpsilonGreedy(k=k, epsilon=0.1), UpperConfidence(k=k, c=c)]
    avg_rewards, _, _ = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_reward_plot(axis, avg_rewards, bandits)
    plt.show()


def figure_2_5(k=10, repeats=1000, steps=1000):
    env = TestBed(k=k, baseline=4)
    bandits = [
        GradientBandit(k=k, stepsize=0.1),
        GradientBandit(k=k, stepsize=0.4),
        GradientBandit(k=k, stepsize=0.1, use_baseline=False),
        GradientBandit(k=k, stepsize=0.4, use_baseline=False),
    ]
    _, avg_optact, _ = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_optact_plot(axis, avg_optact, bandits)
    plt.show()


def playground(k=10, repeats=2000, steps=1000):
    """ Function for if you want to play around with bandits"""
    env = TestBed(k=k, baseline=0)
    bandits = [
        # EpsilonGreedy(k=k, epsilon=0.1, stepsize=0.1),
        # GradientBandit(k=k, stepsize=0.1),
        # EpsilonGreedy(k=k, initial_value=5, epsilon=0.1),
        # UpperConfidence(k=k, c=0.5),
        # UpperConfidence(k=k, c=1),
        # ThompsonSampling(k=k, m_a=0, v_a=0.05, alpha_a=0.05, beta_a=0.05),
        # ThompsonSampling(k=k, m_a=0, v_a=0.5, alpha_a=0.5, beta_a=0.5),
        # ThompsonSampling(k=k, m_a=0, v_a=1, alpha_a=1, beta_a=1),
        # UpperConfidence(k=k, c=2),
        # GradientBandit(k=k, stepsize=0.01),
        # GradientBandit(k=k, stepsize=0.1),
        # GradientBandit(k=k, stepsize=0.5),
    ]

    avg_reward, avg_optact, avg_regret = run_experiment(
        env, bandits, repeats, steps
    )

    fig, axes = plt.subplots(3, 1)
    make_reward_plot(axes[0], avg_reward, bandits)
    make_optact_plot(axes[1], avg_optact, bandits)
    make_regret_plot(axes[2], avg_regret, bandits)
    plt.show()


def main():
    # enable or disable plots you want to see
    # figure_2_2()
    # figure_2_3()
    # figure_2_4()
    # figure_2_5()
    playground(repeats=1000, steps=2000)


if __name__ == "__main__":
    main()
