#!/bin/bash
export TOKENIZERS_PARALLELISM=false
python imdb.py \
	--gradient_clip_val 1.0 \
	--max_epochs 1 \
	"$@"
