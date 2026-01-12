#!/bin/bash

python -u ../architector_dataset_generate_structures.py --input_file input.json --n_cores 4 2>&1 | tee log.txt
