# Makefile for Thompson Sampling experiments
#
# Author: G.J.J. van den Burg
# Copyright (c) 2020 G.J.J. van den Burg
# License: MIT
# Date: 2020-02-09
#

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS += --no-builtin-rules

OUTPUT_DIR=./output

TARGETS = $(OUTPUT_DIR)/hyperparam.json \
	  $(OUTPUT_DIR)/compare_ucb.json \
	  $(OUTPUT_DIR)/hyperparam_mini.json \
	  $(OUTPUT_DIR)/compare_ucb_mini.json \
	  $(OUTPUT_DIR)/ucb_bounds.json

.PHONY: all copy clean

all: $(TARGETS)

output-dir:
	mkdir -p $(OUTPUT_DIR)

hyperparam: $(OUTPUT_DIR)/hyperparam.json $(OUTPUT_DIR)/hyperparam_mini.json

compare_ucb: $(OUTPUT_DIR)/compare_ucb.json $(OUTPUT_DIR)/compare_ucb_mini.json

$(OUTPUT_DIR)/hyperparam.json: thompson_figure.py k_armed_testbed.py | output-dir
	python $< --output $@ hyperparam

$(OUTPUT_DIR)/hyperparam_mini.json: sample.py $(OUTPUT_DIR)/hyperparam.json | output-dir
	python $< --sample 10 hyperparam.json $@

$(OUTPUT_DIR)/compare_ucb.json: thompson_figure.py k_armed_testbed.py | output-dir
	python $< --output $@ compare_ucb

$(OUTPUT_DIR)/compare_ucb_mini.json: sample.py $(OUTPUT_DIR)/compare_ucb.json | output-dir
	python $< --sample 10 compare_ucb.json $@

$(OUTPUT_DIR)/ucb_bounds.json: ucb_bounds.py | output-dir
	python $< --output $@

clean:
	rm -f $(TARGETS)
