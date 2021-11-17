# Experimental Results

To see if our custom task architecture is able to order the sentences, we run the experiment using the ROCStories Dataset with three different pretrained language models.
Also, we run the experiment using the same set of parameters with each framework to check if they perform consistently or any under-the-hood magic influences the scores.

## Hyperparemters

We employ `distilbert-base-cased`, `bert-base-cased`, and `roberta-base` as pretrained models.
Intuitively, we suspect `roberta-base` to perform best, with `bert-case` reaching a tight second place and `distilbert` to fall short behind the large two models.
We use the AdamW optimizer with a learning rate of $3e-5$. We finetune for $3§ epochs and validate our models on the test set while the validation set is only used for tracking the progress during training.
Furthermore, we use the same random seed across all models and frameworks. Due to varying model sizes, we use different batch sizes to fit the model on the GPU. To ensure that the batch sizes do not affect the performance, we use gradient accumulation to ensure that each model makes the same number of backward steps.
In addition to these basic parameters, each framework has a set custom parameter that we leave untouched and use the default configuration.

## Results

```

```