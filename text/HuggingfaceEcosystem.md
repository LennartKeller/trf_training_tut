# The Huggingface ecosystem

In 2018 on the same day that Google published its research implementation of BERT, developed in Tensorflow, Thomas Wolf, a researcher at the NLP startup Huggingface, created a Github repository called "PyTorch-transformers." The goal of this project was to load the weights of the pretrained BERT model in a PyTorch model.


From here on, this repository quickly evolved into the Transformers library, which sits at the core of the Huggingface NLP infrastructure. The goal of the transformers library is to provide the majority of transformer-based neural language models alongside all of the extra tooling required to use them.


On this path the Huggingface team also started to add support for other deep learning frameworks than PyTorch, such as Tensorflow or the newly created JAX library. But these features are relatively new and subject to frequent significant changes, so that this work will only focus on the much more stable PyTorch branch of the Transformers library.


A notable characteristic of these models is that they all require a custom tokenizer. The Tokenizers library from Huggingface provides these.
To complete the toolset of the NLP pipeline, Huggingface also published a library for Dataset management, called Datasets.


With these three libraries, it is possible to cover the overwhelming majority of possible tasks.


But relying on PyTorch as the underlying deep learning framework comes with one caveat: Unlike Tensorflow, which has integrated Keras as a high-level API for training neural networks, PyTorch does not provide any function to facilitate the training process.
Due to PyTorch's research-orientated nature, it is entirely up to the users to implement the training process. While this is useful when researching and experimenting while developing new techniques, it is instead time-consuming when applying standard models to standard tasks like text classification.
In this case, it often comes down to a loop like:

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

Not only can it become quite tedious to write this loop (or variations of it) for various projects, but more gravely, it sets a barrier of entry for beginners or non-experts because it adds another layer of complexity when tinkering around with deep learning.

Another implication of outsourcing this process to the users hits when the models grow in size. Modern language models require a massive amount of memory even when trained with tiny batch sizes. There are strategies to overcome these limitations, like gradient accumulation. But all these tricks again have to be implemented by the user.
While one can argue that most of these tweaks are pretty easy to implement, and there is a vast number of educational material available, the downside comes very clear when working with models that do not even fit on a single GPU. These models have to be trained in a distributed manner across multiple devices. When doing so, the training loop itself gets very complex and challenging to implement.
To overcome these limitations, various frameworks aim to take over the training for the user.
They all promise to lower the complexity of the training process by taking critical parts away from the user's hand.