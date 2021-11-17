#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"

printf "\n\nRunning Huggingface Trainer\n\n"
printf "\n\nBert\n\n"
(cd huggingface-trainer && bash run_rocstories_bert.sh)
printf "\n\nRoberta\n\n"
(cd huggingface-trainer && bash run_rocstories_roberta.sh)
printf "\n\nDistilbert\n\n"
(cd huggingface-trainer && bash run_rocstories_distilbert.sh)

printf "\n\nRunning Pytorch Lightning\n\n"
printf "\n\nBert\n\n"
(cd pytorch-lightning && bash run_rocstories_bert.sh)
printf "\n\nRoberta\n\n"
(cd pytorch-lightning && bash run_rocstories_roberta.sh)
printf "\n\nDistilbert\n\n"
(cd pytorch-lightning && bash run_rocstories_distilbert.sh)

printf "\n\nRunning Poutyne\n\n"
printf "\n\nBert\n\n"
(cd poutyne && python rocstories_bert.py)
printf "\n\nRoberta\n\n"
(cd poutyne && python rocstories_roberta.py)
printf "\n\nDistilbert\n\n"
(cd poutyne && python rocstories_distilbert.py)