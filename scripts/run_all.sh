#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

printf "\n\nRunning Huggingface Trainer\n\n"
printf "\nnBert\nn"
(cd huggingface-trainer && bash run_rocstories_bert.sh)
printf "\Roberta\nn"
(cd huggingface-trainer && bash run_rocstories_roberta.sh)
printf "\Distilbert\nn"
(cd huggingface-trainer && bash run_rocstories_distilbert.sh)

printf "\n\nRunning Pytorch Lightning\n\n"
printf "\nnBert\nn"
(cd pytorch-lightning && bash run_rocstories_bert.sh)
printf "\Roberta\nn"
(cd pytorch-lightning && bash run_rocstories_roberta.sh)
printf "\Distilbert\nn"
(cd pytorch-lightning && bash run_rocstories_distilbert.sh)