# Save And Resume your Experiments

This repo contains the code to show how to save checkpoints during training and resume your experiments from them.
We will show you how to perform it on Tensorflow, Keras and PyTorch.

## Why checkpointing?

![save game screen FF-like](https://i.imgur.com/xdpSAzq.png)

Image your experiments as a video game, sometimes you want to save your game or resume it from an existing state. Checkpoints in Machine/Deep Learning experiments are the same thing, you do not want to lose your experiments due to blackout, OS faults or other types of bad errors. Sometimes you want just to resume a particular state of the training for new experiments or try different things. That's why you need checkpoints!

Not to mention that without a checkpoint at the end of the training, you will have lost all the training!  Like finishing a game without saving at the end.

## Checkpoint Strategies

There are different checkpoint strategies according to the type of training regime you are performing:

- Short Training Regime (minutes - hours)
- Normal Training Regime (hours - day)
- Long Training Regime (days - weeks)

### Short Training Regime
In this type of training regime is a common practice to save only a checkpoint at the end of the training.

### Normal Training Regime
In this type of training regime is a common practice to save multiple checkpoints every n_epochs and keep track about what's the best one with respect to validation metric we care about. Usually there is a fixed number of checkpoints we care about so to not take to much space, such as restrict it to keep only 10 checkpoints(the new ones will replace the last ones).

### Long Training Regime
In this type of training regime is a common practice to save multiple checkpoints every n_epochs and keep track about what's the best one with respect to validation metric we care about. Since the training can be really long, is common to save less frequently but keep more checkpoints file, so that we will be able to resume the training in particular situations.

*Obviously you can use a custom Checkpoint Strategy according to your need and the task you will run.*

## The Tradeoff

The tradeoff is between the **frequency** and the **number of checkpoints files** to keep. Let's take a look what's happen when we act over these two parameters:

Frequency | Number of checkpoints to keep | Cons | Pro
--------- | ----------------------------- | ---- | ---
High | High | You need a lot of space!! | You can resume very quickly in almost all the interesting training states.
High | Low | You could have lost preciuos states. | Minimize the storage space you need.
Low | High | If some things happened between two checkpoints, it will cost you some time to retrieve it. | You can resume the experiments in a lot of interesting states.
Low | Low | You could have lost preciuos states | Minimize the storage space you need.


Now you have a good intuition about what's the best strategy you can adopt according to your training regime.

## Save and Resume on FloydHub
For this example we use the Deep Learning hello-world: the [MNIST](http://yann.lecun.com/exdb/mnist/) classification task using a Convolutional Neural Network model.

### Tensorflow
Soon.

### Keras

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

Keras provide an amazing API for saving and loading a checkpoints. Let's take a look:

#### Saving
Keras provides a set of functions called [callback](https://keras.io/callbacks/): you can think of it as events that will triggered at certain training state. The callback we need for checkpointing is the [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) which provides all the features we need according to the checkpoint strategy adopted.

**This function save only the model's weights**, if you want to save the whole model or some of the components take a look at [how can i save a keras model from Keras docs](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).

First of all we have to import the callback functions:
```python
from keras.callbacks import ModelCheckpoint
```
Next, just before the call to `model.fit(...)` it's time to prepare the checkpoint strategy.

```python
# Checkpoint In the /output folder
filepath = "/output/mnist-cnn-best.hdf5"

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
```
- `filepath="/output/mnist-cnn-best.hdf5"`: FloydHub returns only the contents inside the `/output` folder! See [save output in the docs](https://docs.floydhub.com/guides/data/storing_output/),
- `monitor='val_acc'`: the metric we care about, validation accuracy,
- `verbose=1`: it will print more infos,
- `save_best_only=True`: Keep only the best one(in term of max val_acc),
- `mode='max'`: save the one with max validation accuracy.

Default period(checkpointing frequency) is set to 1, this means at the end of every epoch.

For more infos(such as filepath formatting options, checkpointing period and more) we encourage you to explore the [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) API.

Now we are ready to see it apply during training, to do this, we need to pass the callback variable to the `model.fit(...)` call:

```python
# Train
model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(x_test, y_test),
                callbacks=[checkpoint])  # <- Apply our checkpoint strategy
```

According to the chosen strategy you will see:
```
# This line when the training reach a new max
Epoch <n_epoch>: val_acc improved from <previus val_acc> to <new max val_acc>, saving model to /output/mnist-cnn-best.hdf5

# Or this line
Epoch <n_epoch>: val_acc did not improve
```

That's it about saving a checkpoint in Keras.

#### Resuming
Keras models have the [`load_weights()`](https://github.com/fchollet/keras/blob/master/keras/models.py#L718-L735) method which load the weights from a hdf5 file.

To load the model's weight you have to add this line just after the model definition:

```python
... # Model Definition

model.load_weights(resume_weights)
```

That's it about resuming a checkpoint in Keras.

This tutorial follow a basic setup, if you have a more sofisticated experiments you will have to hack it a bit.

### PyTorch

![Pytorch logo](http://pytorch.org/docs/master/_static/pytorch-logo-dark.svg)

Unfortunately at the moment PyTorch has not a great API as Keras, therefore we need to write our own solution according to the checkpoint strategy adopted(the same we have used on Keras).


#### Saving
saving bla bla bla

#### Resuming
resuming bla bla bla



Have a great training :)

## Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me, open an issue or a PR!


