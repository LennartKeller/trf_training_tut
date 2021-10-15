# Comparison

In the following sections, we implement the sentence ordering task with each framework.

While doing so, we have a list of requirements that each framework should fulfill.
These requirements can further be divided into two groups. Mandatory requirements are necessary for a 


## Mandatory requirements

__Tracking__

We want to track the progress our model makes throughout the training. Ideally, the model should be validated every 500 training steps, using the loss function and the other two metrics.

__Checkpointing__

Throughout, the training we want to save checkpoints of our model. Similar to the tracking, we want to save our model every 1000 training steps.

__Multi-GPU Training__

Since most language models come in different sizes, we test the capability to do Multi-GPU training out-of-the-box.

__Seeding__

Recent research has shown that randomly initialized parameters (CITE).
So we want to be able to control this randomness using a fixed random seed.




