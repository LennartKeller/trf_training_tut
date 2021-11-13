# Poutyne

Compared to the other two frameworks, Poutyne has a different scope.

Instead of trying to make the training of a fixed set of models as easy as possible like Huggingface `Trainer`, or facilitating the creation and training of custom models like PyTorch Lightning, it tries to bring the ease of the Keras API from the realms of Tensorflow to the world of PyTorch.
The benefits of the Keras API are its simplicity and orientation at well-established machine learning frameworks like Scikit-Learn.
This simplicity lowers the barrier of entry for beginners because it lowers the amount of time needed to get hands-on training for their first model.
The following exemplary listing shows the typical workflow in Poutyne.

```python
from poutyne import Model

...

network = make_network()
X_train, y_train = load_data(subset="train")
X_val, y_val = load_data(subset="validation")
X_test, y_test = load_data(subset="test")

model = Model(
    network,
    "sgd",
    "cross_entropy",
    batch_metrics=["accuracy"],
    epoch_metrics=["f1"],
    device="cuda:0"
)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=64
)

results = model.evaluate(X_test, y_test, batch_size=128)
```

Like Keras, Poutyne automates many steps for standard cases like the configuration of the optimizer or the loss function.
However, Poutyne does not mimic the whole Keras API but only the training part.
The model's creation still has to be done in plain PyTorch, which is generally a bit trickier than Keras because the dimensions of all layers have to be chosen manually.
In addition to the training functions, Poutyne also provides utilities to conduct and save whole experiments and utilities for creating checkpoints, logging, and learning rate schedule.

## Classes

### Model

The `Model` class is intended to handle the training of any neural network
Technically, it wraps a neural network alongside an optimizer, loss function, and validation metrics.
To train the model, it exposes different variants of the `fit`-method, each of which can process the training data in another format.
The standard `.fit`-methods expects the data as a list of batches, while the `fit_dataset` method can directly work on PyTorch `Datasets`.
The `fit_generator` can operate on generators that yield the data batch by batch as a third option.
In addition to that, the `fit`-methods also receive other hyperparameters to control the training.

The `evaluate`-method computes the loss and all other metrics on unseen data without doing backpropagation.
If only the predictions of the network are needed, the `predict`-method can be used.
Similar to the variations of the `fit`-methods, these methods are offered in different versions too.

* Wraps the model and optimizer
* Contains the loss function and validation metrics
* Models are fitted using the fit method
  * FIT  the data for training und validation
  * FIT => ON DATALOADER
  * FIT => ON TORCH DATASET
  * FIT_GENERATOR => ON A GENERATOR
* PREDICT => RETURNS PREDICTION
* EVALUATE => COMPUTES LOSS AND METRICS FOR THE MODEL WITHOUT BACKPROP

### Experiment

The `Experiment` class is an extended version of the `Model` class that comes with helpful additions for conducting deep learning experiments.
Like the `Model` class, an `Experiment` is equipped with the neural network, optimizer, loss function, and metrics into a single object and has methods to start the training, evaluation, or prediction.
In contrast to the `Model` class, which is solely designed to train a model, the `Experiment` class provides additional features to organize and track the training
For example, it supports logging the progress to various formats, like a CSV table or Tensorboard.
Monitoring allows the `Experiment` class to save checkpoints of the model that perform best with respect to one of the validation metrics.
Also, it saves all the intermediate results and tracked values to the disk.

For the two primary task types, classification and regression, the experiment automatically configures all metrics and the loss function if these tasks are specified in the `task` parameter when initializing the `Experiment`.

### Data

Poutyne is data agnostic meaning, that it does not provide any tooling to load, process, and store the training data.
The only requirements are that the data comes in one of the supported formats and that each batch consists of two objects: one that holds the training data and one that contains the label.
To compare the model's output with the labels, it has to be in the same format.

## Additional Features

#### Metrics

Poutyne has a custom API for implementing metrics.
It distinguishes between two types of metrics, batch metric and epoch metrics.
Batch metrics are computed per batch and averaged to obtain the results for one single epoch.
Epoch metrics are computed on the gathered results of one entire epoch. Thus, they are a good choice for measures that would suffer from averaging over the batch results, like the F-score.
Poutyne provides predefined metrics for both types. But, unfortunately, they only cover classification tasks.
There are two options to add other metrics. Either they have to be implemented manually or taken from Scikit-Learn and made compatible using a built-in wrapper class.
Metrics are passed to `Model` or `Experiment` while their initialization.
#### Callbacks

Callbacks are intended to extend the functions of the `Model` or `Experiment` class. Like the callbacks from the other frameworks, they have access to the model's current state and can perform actions at various steps while training.
There are many predefined callbacks available that perform all kinds of tasks, ranging from logging, keeping track of gradients, scheduling the learning, creating checkpoints, to sending notifications to inform clients about the progress of the training.

## Implementation

### Model

To be able to use a `transformers` model with Poutyne it has to be able to rec

### Loss

### Data


### Complete code

## Conclusion

* Not fair to compare to other ones
* Suitable for beginners