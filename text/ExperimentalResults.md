# Experimental Results

To see if our custom task architecture is able to order the sentences, we run the experiment using the ROCStories Dataset with three different pretrained language models.
Also, we run the experiment using the same set of parameters with each framework to check if they perform consistently or if any under-the-hood magic influences the scores.

## Hyperparemters

We employ `distilbert-base-cased`, `bert-base-cased`, and `roberta-base` as pretrained models.
Intuitively, `roberta-Base` to perform best, with `bert-case` reaching a tight second place and `distilbert` to fall short behind the large two models.
We use the AdamW optimizer with a learning rate of $3e-5$. We finetune for $3§ epochs and validate our models on the test set while the validation set is only used for tracking the progress during training.
Furthermore, we use the same random seed across all models and frameworks. Due to varying model sizes, we use different batch sizes to fit the model on the GPU. To ensure that the batch sizes do not affect the performance, we use gradient accumulation to ensure that each model makes the same number of backward steps.
In addition to these basic parameters, each framework has a set custom parameter that we leave untouched and use the default configuration.

## Results

```{list-table}
:header-rows: 1
:name: results-table

* - Framework
  - Model
  - Loss
  - Accuracy
  - Kendall's Tau
* - Huggingface `Trainer`
  - `bert-base-cased`
  - 2.447
  - 0.699
  - 0.792
* -
  - `roberta-base`
  - 1.619
  - 0.786
  - 0.861
* -
  - `distilbert-base-cased`
  - 2.800
  - 0.653
  - 0.753
* - PyTorch Lightning
  - `bert-base-cased`
  - 2.621
  - 0.696
  - 0.785
* -
  - `roberta-base`
  - 1.755
  - 0.776
  - 0.853
* -
  - `distilbert-base-cased`
  - 2.933
  - 0.651
  - 0.748
* - Poutyne
  - `bert-base-cased`
  - 2.714
  - 0.687
  - 0.776
* -
  - `roberta-base`
  - 2.106
  - 0.748
  - 0.830
* -
  - `distilbert-base-cased`
  - 3.024
  - 0.645
  - 0.742
```

{numref}`results-table` shows the results of the runs. As expected, `roberta-base` achieved the best results. However, the margin by which it outperforms the standard `bert-base-cased` model is surprising. In contrast to Bert models, Roberta models do not use the next sentence prediction objective during their finetuning stages. Next sentence prediction is the task of deciding if two consecutive sentences in the input sequence are natural successors or not. Intuitively, the knowledge obtained from this task should help order sentences too. Obviously, without additional inspection, each further presumption is mere speculation. Still, it seems like the fact that Roberta models are pretrained on a larger dataset outweighs the missing pretraining on the sentence level.