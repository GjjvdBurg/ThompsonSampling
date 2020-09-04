#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample a LineGraph JSON data file

This is just a convenience script to reduce the filesize of the output of 
thompson_figure.py

Author: Gertjan van den Burg
License: See LICENSE file
Copyright: 2020, G.J.J. van den Burg

"""

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--sample", help="Number of values to skip", type=int, default=10
    )
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("output", help="Output JSON file")
    return parser.parse_args()


def sample(values, k):
    new = []
    for i, x in enumerate(values):
        if i % k == 0:
            new.append(x)
    return new


def main():
    args = parse_args()
    with open(args.input, "r") as fp:
        data = json.load(fp)

    meta = data["_meta"]
    X = data["data"]["X"]
    series = data["data"]["series"]

    Xs = sample(X, args.sample)
    Ss = [
        dict(name=S["name"], values=sample(S["values"], args.sample))
        for S in series
    ]
    data = dict(X=Xs, series=Ss)

    sampled = dict(_meta=meta, data=data)

    with open(args.output, 'w') as fp:
        json.dump(sampled, fp)


if __name__ == "__main__":
    main()
