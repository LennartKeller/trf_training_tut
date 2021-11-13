#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

printf "\n\nRunning Huggingface Trainer\n\n"
(cd huggingface-trainer && python rocstories.py)

printf "\n\nRunning Poutyne\n\n"
(cd poutyne && python rocstories.py)

printf "\n\nRunning Pytorch Lightning\n\n"
(cd pytorch-lightning && python rocstories.py)