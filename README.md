# Save And Resume your Experiments

This repo contains the code to show how to save checkpoints during training and resume your experiments from them.
We will show you how to perform it on Tensorflow, Keras and PyTorch.

## Why checkpointing?

![save game screen FF-like](https://i.imgur.com/xdpSAzq.png)

Image your experiments as a video game, sometimes you want to save your game or resume it from an existing state. Checkpoints in Machine/Deep Learning experiments are the same thing, you do not want to lose your experiments due to blackout, OS faults or other types of bad errors. Sometimes you want just to resume a particular state of the training for new experiments or try different things. That's why you need checkpoints!

Not to mention that without a checkpoint at the end of the training, you will have lost all the training! Like finishing a game without saving at the end.

## What is a checkpoint made of?

A checkpoint can consist of:

- The architecture of the model, allowing to re-create the model
- The weights of the model
- The training configuration (loss, optimizer, epochs and other meta-infos)
- The state of the optimizer, allowing to resume training exactly where you left off.

*Taken from Keras docs [how-can-i-save-a-keras-model](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)*.

## Checkpoint Strategies

There are different checkpoint strategies according to the type of training regime you are performing:

- Short Training Regime (minutes - hours)
- Normal Training Regime (hours - day)
- Long Training Regime (days - weeks)

### Short Training Regime
In this type of training regime is a common practice to save only a checkpoint at the end of the training or at the end of every epoch.

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

Before you start, log in on FloydHub with the [floyd login](http://docs.floydhub.com/commands/login/) command, then fork and init the project:

```bash
$ git clone https://github.com/floydhub/save-and-resume.git
$ cd save-and-resume
$ floyd init save-and-resume
```

For this examples we use the Deep Learning hello-world: the [MNIST](http://yann.lecun.com/exdb/mnist/) classification task using a Convolutional Neural Network model.

The strategy we have adopted for the next example is the following:
- Keep only one checkpoints
- Trigger the strategy at the end of every epoch
- Save the one with the best(max) validation accuracy

Considering the toy example, a Short Training Regime provide a good strategy.

*As said this tutorial follows a basic setup, if you have a more sofisticated experiments you will have to hack it.*

This is the basic template you have to follow for saving and resuming when you run your experimets on FloydHub *via script*:

#### Saving Template command

```bash
floyd run \
    [--gpu] \
    --env <env> \
    --data <your_dataset>:<mounting_point_dataset> \
    "python <script_and_parameters>"
```

The checkpoint of this script must be saved in the `/output` foler.

#### Resuming Template after training

```bash
floyd run \
    [--gpu] \
    --env <env> \
    --data <your_dataset>:<mounting_point_dataset> \
    --data <output_of_previuos_job>:<mounting_point_model> \
    "python <script_and_parameters>"
```
The scipt will resum the checkpoint from the previus Job's Output.

Let's see how to make it tangible for the different framework on FloydHub.

## Tensorflow
![TF logo](https://www.tensorflow.org/_static/268f0926ba/images/tensorflow/lockup.png)

Tensorflow provide different way for saving and resuming a checkpoint. In the example we will use the [tf.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator) API, that behind the scene uses [tf.train.Saver](https://www.tensorflow.org/api_docs/python/tf/train/Saver), [tf.train.CheckpointSaverHook](https://www.tensorflow.org/api_docs/python/tf/train/CheckpointSaverHook) tf.[saved_model.builder.SavedModelBuilder](https://www.tensorflow.org/api_docs/python/tf/saved_model/builder/SavedModelBuilder).

More in detail, it uses the first function to save, the second one to act according to the adopted strategy and the last one to export the model to be served with `export_savedmodel()` method.

### Saving

Before init an Estimator, we have to define the checkpoint strategy. To do this we have to create a configuration for the Estimator using the [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig) API such this:

```python
# Checkpoint Strategy configuration
run_config = tf.estimator.RunConfig(
    model_dir=filepath,
    keep_checkpoint_max=1)
```

In this way we are telling the estimator in which directory save or resume a checkpoint and how many checkpoints to keep.

Then we have to provide it, at the initialization of the Estimator:

```python
# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, config=run_config)
```

That's it about saving a checkpoint in Tensorflow using Estimator.

### Resuming

After having configurated the Estimator, everything is done. If it will find a checkpoint inside the given model folder, it will load the last one.

That's it about resuming a checkpoint in Tensorflow using Estimator.

### Run on FloydHub
Here's the steps to run the example on FloydHub.

#### Via script

First time training:

```bash
floyd run \
    --gpu \
    --env tensorflow-1.3 \
    --data redeipirati/datasets/mnist/1:input \
    'python tf_mnist_cnn.py'
```

- The `--env` flag specifies the environment that this project should run on, which is Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6.
- The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.


#### Via Jupyter

## Keras

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

Keras provide a great API for saving and loading a checkpoints. Let's take a look:

### Saving
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

### Resuming
Keras models have the [`load_weights()`](https://github.com/fchollet/keras/blob/master/keras/models.py#L718-L735) method which load the weights from a hdf5 file.

To load the model's weight you have to add this line just after the model definition:

```python
... # Model Definition

model.load_weights(resume_weights)
```

That's it about resuming a checkpoint in Keras.


### Run on FloydHub
Here's the steps to run the example on FloydHub.

#### Via script

First time training:

```bash
floyd run \
    --gpu \
    --env tensorflow-1.3 \
    'python keras_mnist_cnn.py'
```

- The `--env` flag specifies the environment that this project should run on, which is Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6.
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.

[Keras provide an API to handle MNIST data](https://keras.io/datasets/#mnist-database-of-handwritten-digits), so we can skip the dataset mounting since the dataset size is irrilevant.

Resuming:

```bash
floyd run \
    --gpu \
    --env tensorflow-1.3 \
    --data <your-username>/projects/save-and-resume/<jobs>/output:/model \
    'python keras_mnist_cnn.py'
```

- The `--env` flag specifies the environment that this project should run on, which is Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6.
- The `--data` flag specifies that the output of a previus Job should be available at the `/model` directory
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.


#### Via Jupyter

```bash
floyd run \
    --gpu \
    --env tensorflow-1.3 \
    --mode jupyter
```

- The `--env` flag specifies the environment that this project should run on, which is Tensorflow 1.3.0 + Keras 2.0.6 on Python3.6.
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.
- The `--mode` flag specifies that this job should provide us a Jupyter notebook.

Add `--data <your-username>/projects/save-and-resume/<jobs>/output:/model`, if you want to load a checkpoint from a previous Job.


## PyTorch

![Pytorch logo](http://pytorch.org/docs/master/_static/pytorch-logo-dark.svg)

Unfortunately at the moment PyTorch has not a great API as Keras, therefore we need to write our own solution according to the checkpoint strategy adopted(the same we have used on Keras).


### Saving
PyTorch does not provide an all-in-one API in which defines the checkpoint strategy but it provide a simple way to save and resume a checkpoint. According the official docs about [semantic serialization](http://pytorch.org/docs/master/notes/serialization.html), the best practice consist of save only the weights due to code refactoring issue.

Let's take a look at how to save the model weights in PyTorch:


First of all define a `save_checkpoint` function which handles all the instructions about the number of checkpoints to keep and the serialization on file:

```python
def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")
```

Then, inside the training(usually a for loop with the number of epochs), we define the checkpoint frequency(at the end of every epoch) and the informations(epochs, model weights and best accuracy achieved) we want to save:

```python
...

# Training the Model
for epoch in range(num_epochs):
    train(...)  # Train
    acc = eval(...)  # Evaluate after every epoch

    # Some stuff with acc(accuracy)
    ...

    # Get bool not ByteTensor
    is_best = bool(acc.numpy() > best_accuracy.numpy())
    # Get greater Tensor to keep track best acc
    best_accuracy = torch.FloatTensor(max(acc.numpy(), best_accuracy.numpy()))
    # Save checkpoint if is a new best
    save_checkpoint({
        'epoch': start_epoch + epoch + 1,
        'state_dict': model.state_dict(),
        'best_accuracy': best_accuracy
    }, is_best)
```

That's it about saving a checkpoint in PyTorch.

### Resuming
To resume a checkpoint, before the training we have to load the weights and the meta information we need:

```python
# cuda = torch.cuda.is_available()
if cuda:
    checkpoint = torch.load(resume_weights)
else:
    # Load GPU model on CPU
    checkpoint = torch.load(resume_weights,
                            map_location=lambda storage,
                            loc: storage)
start_epoch = checkpoint['epoch']
best_accuracy = checkpoint['best_accuracy']
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))
```

For more info about loading GPU trained weights on CPU, see this [PyTorch discussion](https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032).

That's it about resuming a checkpoint in PyTorch.

### Run on FloydHub
Here's the steps to run the example on FloydHub.

#### Via script

First time training:

```bash
floyd run \
    --gpu \
    --env pytorch-0.2 \
    --data redeipirati/datasets/pytorch-mnist/1:input \
    'python pytorch_mnist_cnn.py'
```

- The `--env` flag specifies the environment that this project should run on, which is a PyTorch 0.2.0 on Python 3.
- The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.


Resuming:

```bash
floyd run \
    --gpu \
    --env pytorch-0.2 \
    --data redeipirati/datasets/pytorch-mnist/1:input \
    --data <your-username>/projects/save-and-resume/<jobs>/output:/model \
    'python pytorch_mnist_cnn.py'
```

- The `--env` flag specifies the environment that this project should run on, which is a PyTorch 0.2.0 on Python 3.
- The first `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
- The second `--data` flag specifies that the output of a previus Job should be available at the `/model` directory
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.

#### Via Jupyter

```bash
floyd run \
    --gpu \
    --env pytorch-0.2 \
    --data redeipirati/datasets/pytorch-mnist/1:input \
    --mode jupyter
```

- The `--env` flag specifies the environment that this project should run on, which is a PyTorch 0.2.0 on Python 3.
- The `--data` flag specifies that the pytorch-mnist dataset should be available at the `/input` directory
- Note that the `--gpu` flag is optional for now, unless you want to start right away to run the code on a GPU machine.
- The `--mode` flag specifies that this job should provide us a Jupyter notebook.

Add `--data <your-username>/projects/save-and-resume/<jobs>/output:/model`, if you want to load a checkpoint from a previous Job.

Have a great training :)

## Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me, open an issue or a PR!


