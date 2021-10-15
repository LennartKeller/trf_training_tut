---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: ["remove-cell"]

from datasets import set_caching_enabled
set_caching_enabled(False)

# import pprint
# pp = pprint.PrettyPrinter(depth=6, compact=True)
# print = pp.pprint
```

# The Huggingface ecosystem


## `tranformers`

In 2018 on the same day that Google published its research implementation of BERT, developed in Tensorflow, Thomas Wolf, a researcher at the NLP startup Huggingface, created a Github repository called "PyTorch-transformers." The goal of this project was to load the weights of the pretrained BERT model in a PyTorch model.


From here on, this repository quickly evolved into the Transformers library, which sits at the core of the Huggingface NLP infrastructure. The goal of the transformers library is to provide the majority of transformer-based neural language models alongside all of the extra tooling required to use them.


On this path the Huggingface team also started to add support for other deep learning frameworks than PyTorch, such as Tensorflow or the newly created JAX library. But these features are relatively new and subject to frequent significant changes, so that this work will only focus on the much more stable PyTorch branch of the Transformers library.

## `tokenizers`

A notable characteristic of these models is that they all ship with a custom tokenizer. These tokenizers are subword tokenizers that were fitted to represent the vocabulary of the training data of the models with a fixed size vocabulary.
Huggingface provides another library called `tokenizers`. It provides the a general framework to load all the tokenizers of each model.

## `datasets`

Lastly, to complete the toolset of a transformers-NLP pipeline, Huggingface also develops a library for Dataset management, called Datasets.

With these three libraries, it is possible to cover the overwhelming majority of possible tasks.

## Interoperability
To make all these libraries as interoperable as possible, the standard data exchange format is a dictionary that contains all the argument names of the function or method that is supposedly called next as keys and the data as values.

```{code-cell} ipython3
from transformers import AutoTokenizer, AutoModel
from datasets import Dataset

model = AutoModel.from_pretrained("bert-base-cased", add_pooling_layer=False)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

dataset = Dataset.from_dict({"text": ["Dictionaries? Everywhere!"]})

data = dataset[0]
print(data)

inputs = tokenizer(data["text"], return_tensors="pt")
outputs = model(**inputs)
print(outputs)
```


## `PyTorch`-Backend

Relying on PyTorch as the underlying deep learning framework comes with one caveat: Unlike Tensorflow, which has integrated Keras as a high-level API for training neural networks, PyTorch does not provide any tools to facilitate the training process.
Due to PyTorch's research-orientated nature, it is entirely up to the users to implement the training process. While this is no problem when researching and experimenting with new techniques, it is time-consuming in the practitioner's case.
When applying standard models to tasks like text classification, implementing the training loop is an obstacle that only increases development time. Also, it introduces a new space for making errors.

In most application-oriented scenarios, the training loop roughly looks like this:

```python
...
model = create_model()
train_data, val_data = load_data()
optimizer = torch.optim.SGD(lr=5e-5, params=model.parameters())
for train_step, batch in enumerate(train_data):
    input_data, targets = batch
    input_data = input_data.to(DEVICE)
    targets = targets.to(DEVICE)
    outputs = model(input_data)
    loss = loss_function(outputs, targets)
    # Compute gradients w.r.t the input data
    loss.backward() 
    # Update the parameters of the model
    optimizer.step() 
    # Clear the gradients before next step
    optimizer.zero_grad() 
    train_log(train_step, loss)
    # Validate the performance of the model every 100 train steps
    if train_step % 100 == 0:
        for val_step, batch in enumerate(val_data):
                input_data, targets = batch
                input_data = input_data.to(DEVICE)
                targets = targets.to(DEVICE)
            with torch.no_grad():
                outputs = model(input_data)
                val_loss = loss_function(outputs, targets).detach().cpu()
                # Compute other val metrics (i.e. accuracy)
                val_score = other_metric(outputs, targets)
                val_log(val_step, val_loss, val_loss)
...
```

But not only can it become quite tedious to write this loop (or variations of it) for various projects, but more gravely, it sets a barrier of entry for beginners or non-experts because it adds another layer of complexity when tinkering around with deep learning.

Another implication of outsourcing this process to the users hits when the models grow in size. Modern language models require a massive amount of memory even when trained with tiny batch sizes. There are strategies to overcome these limitations, like gradient accumulation. But all these tricks again have to be implemented by the user.
While one can argue that most of these tweaks are pretty easy to implement, and there is a vast number of educational material available, the downside comes very clear when working with models that do not even fit on a single GPU. These models have to be trained in a distributed manner across multiple devices. When doing so, the training loop itself gets much more complex and challenging to implement.
Various frameworks aim at streamlining the training of neural networks for the user.
