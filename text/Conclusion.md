# Conclusion

Due to their different scopes labeling one of the presented frameworks as the "best one" would paint a misleading picture.
Obviously, the built-in `Trainer` of the `transformers` library is optimally aligned with the rest of the library, which facilitates the training of language models in many cases.
This ease, combined with its optimized set of predefined parameters, makes it the best choice when training standard models.
Even in cases like ours, where a standard model is combined with a custom loss, the `Trainer` requires few adaptions.
However, the goal to allow those adaptions with as few lines of code as possible also has some drawbacks. Conceptually, the separation of concerns between integral parts of the model and additional logic is less strict.
For example, by default `transformers` models incorporate the loss function into their heads. But if this loss should be discarded, the custom loss function is bound to the custom subclass of the `Trainer`. This scattering leads to an implicit separation of the model and its loss.
Without access to the `Trainer` subclass, continuing the model's training is impossible. More gravely, this fact is hidden too, since loading the model looks like a standard token classification model and even returns a loss score when fed with the correct data.
Of course, one could argue that it is possible to create proper custom models with an own head, which would be a cleaner way to implement such a model. But since making a custom model is a rather complex process, it would also render the advantage of the `Trainer` irrelevant.

From a conceptual point of view, PyTorch Lightning approach is far more sustainable since its API forces to structure the code into mostly self-contained modules.
While requiring more manual implementation upfront, this approach leads to better code quality and makes models and datasets easily interchangeable and thus more reusable.
However, it also expects the user to have a profound knowledge of PyTorch itself, alongside a profound mental model of how a neural network is trained since there are no shortcuts.
Users who fit these requirements can benefit from PyTorch Lightning's strict specifications.
While these specifications determine the whole process of building neural networks, they do not lose much freedom because the framework is highly flexible and can be modified to a great extend.
Yet, there are drawbacks when working with `transformers` and PyTorch Lightning.
Most notably, the differences in the serialization of models complicate the process of creating checkpoints that can be used interchangeably between PyTorch Lightning and the Huggingface ecosystem.
Also, PyTorch Lightning, like most deep learning frameworks, evolves fast and is updated frequently.
This speed can lead to issues that are hard to understand and fix.
For example, on the machine used to run the experiments of this work, only one of the six available computational backends that govern Multi-GPU training worked reliably.
Using the other ones either led to freezes while training, exceptions that aborted the training complete, or degraded results because the data was corrupted.
These problems were exacerbated because PyTorch code is notoriously hard to debug since most of the computations are done outside the Python runtime and thus hard to access with standard debuggers.
Nonetheless, all the available extension and hyperparameter tuning functions justify the usage of PyTorch Lightning because once the initial setup is running, improving the results is easy and does not require much work.

To be fair, it has to be stated that including Poutyne in this work is unfair since its scope is much more narrow, and, by its intention, it does not try to offer the same set of functions as both other frameworks.
Instead of focusing on its lacks, it becomes clear why Poutyne can be helpful by looking at its conceptual design.
Its conceptual paragon Keras is the most beginner-friendly library to get started with deep learning.
Since Keras dropped support for other backends like PyTorch, this easy is only available for Tensorflow.
However, since most deep learning research is done in PyTorch nowadays, learning Tensorflow has become less attractive because beginners will have to switch from Tensorflow to PyTorch at some point as they progress.
So because learning PyTorch is inevitable for most users, it would be beneficial to be able to start right away with it.
Without being anywhere near as mature as Keras, Poutyne has the potential of offering a real alternative for it in PyTorch.
Additionally, a high-level API that enables the quick training of models effortlessly is also attractive for experienced users who want to test a prototype.
As our results showed, the results won't be as best as possible, but they are sufficient enough to give a first impression.
Like Keras, Poutyne is designed to work best with models built from scratch, but it is easy to adapt the framework to work with pretrained models of all sorts.
The library `poutyne-transformers` might act as a proof-of-concept for further adjustments.

In conclusion, this work showed that all three of the frameworks have a reason for existence. Choosing between PyTorch Lightning and the Hugginface `Trainer` highly depends on the requirements of the project. PyTorch Lightning is a suitable choice for models that are intended to go into production. At the same time, the `Trainer` is a great choice when doing research where the speed of iteration outweighs the sustainability of the code.
Poutyne is an excellent choice for beginners who want to get into deep learning.