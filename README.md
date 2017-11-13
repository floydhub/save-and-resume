# Save And Resume your Experiments

This repo contains the code to show how to save checkpoints during training and resume your experiments from them.
We will show you how to perform it on Tensorflow, Keras and PyTorch.

## Why checkpointing?

![save game screen FF-like](https://i.imgur.com/xdpSAzq.png)

Image your experiments as a video game, sometimes you want to save your game or resume it from an existing state. Checkpoints in ML/DL are the same thing, you do not want to lose your experiments due to blackout, OS faults or other types of bad errors. Sometimes you want just to resume a particular state of the training for new experiments or try different things.

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

## The Tradeoff

The tradeoff is between the **frequency** and the **number of checkpoints files** to keep. Let's take a look what's happen when we act over these two parameters:

Frequency | Number of checkpoints to keep | Cons | Pro
--------- | ----------------------------- | ---- | ---
High | High | You need a lot of space!! | You can resume very quickly in almost all the interesting training states.
High | Low | You could have lost preciuos states. | Minimize the storage space you need.
Low | High | If some things happened between two checkpoints, it will cost you some time to retrieve it. | You can resume the experiments in a lot of interesting states.
Low | Low | You could have lost preciuos states | Minimize the storage space you need.

*Obvsiuolsy you can use a custom Checkpoint Strategy according to your need and the task you will run.*

Now you have a good intuition about what's the best strategy you can adopt according to your training regime.

## Save and Resume on FloydHub

### Tensorflow

### Keras

### PyTorch

Have a great training :)

## Contributing

For any questions, bug(even typos) and/or features requests do not hesitate to contact me, open an issue or a PR!


