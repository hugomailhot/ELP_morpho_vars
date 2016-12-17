# !/usr/bin/env python
# encoding: utf-8

"""
Takes a json file with {"key1":["value1", "value2", ...]} pairs and outputs
a reverse lookup {"value1":"key", "value2":"key"}
"""

import json
import os

project_fp = "/home/hugo/Projects/ELP_morpho_vars"
input_fp = os.path.join(project_fp, "linguistic_data/allomorphs_suffixes.json")
output_fp = os.path.join(project_fp, "linguistic_data/rev_allomorphs_suffixes.json")

with open(input_fp) as f:
    allos = json.load(f)

reverse = {}

for k, values in allos.items():
    for v in values:
        reverse[v] = k

with open(output_fp, 'w') as f:
    json.dump(reverse, f)
