#!/bin/bash

python -u ../architector_dataset_stratified_sampling.py --input_file input.json \
                                                        --n_cores 4 2>&1 | tee log.txt
