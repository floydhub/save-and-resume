"""Convolutional Neural Network Estimator for MNIST, built with tf.layers.

Adapted from:
https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/examples/tutorials/layers/cnn_mnist.py

Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import shutil, os

tf.logging.set_verbosity(tf.logging.INFO)


# Where to save Checkpoint(In the /output folder)
resumepath ="/model/mnist_convnet_model"
filepath = "/output/mnist_convnet_model"

# Hyper-parameters
batch_size = 128
num_classes = 10
num_epochs = 12
learning_rate = 1e-3

# If exists an checkpoint model, move it into the /output folder
if os.path.exists(resumepath):
    shutil.copytree(resumepath, filepath)

# Load training and eval data
mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images  # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print (train_data.shape)
print (eval_data.shape)

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    # Computes 32 features using a 3x3 filter with ReLU activation.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 26, 26, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[3, 3],
      activation=tf.nn.relu)

    # Convolutional Layer #2
    # Computes 64 features using a 3x3 filter.
    # Input Tensor Shape: [batch_size, 26, 26 32]
    # Output Tensor Shape: [batch_size, 24, 24, 64]
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      activation=tf.nn.relu)

      # Pooling Layer
    # Max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 24, 24, 64]
    # Output Tensor Shape: [batch_size, 12, 12, 64]
    pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dropout # 1
    # Add dropout operation; 0.25 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=pool, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 12, 12, 64]
    # Output Tensor Shape: [batch_size, 12 * 12 * 64]
    flat = tf.reshape(dropout, [-1, 12 * 12 * 64])  # 9216


    # Dense Layer # 1
    # Densely connected layer with 128 neurons
    # Input Tensor Shape: [batch_size, 12 * 12 * 64] (batch_size, 9216)
    # Output Tensor Shape: [batch_size, 128]
    dense1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)

    # Dropout # 2
    # Add dropout operation; 0.5 probability that element will be kept
    dropout2 = tf.layers.dropout(
      inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits layer
    # Input Tensor Shape: [batch_size, 128]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout2, units=num_classes)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    # Inference (for TEST mode)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes)
    # Cross Entropy
    loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # AdamOptimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(
              labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=eval_metric_ops)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Checkpoint Strategy configuration
run_config = tf.contrib.learn.RunConfig(
    model_dir=filepath,
    keep_checkpoint_max=1)

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, config=run_config)

# Keep track of the best accuracy
best_acc = 0

# Training for num_epochs
for i in range(num_epochs):
    print("Begin Training - Epoch {}/{}".format(i+1, num_epochs))
    # Train the model for 1 epoch
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=1,
        shuffle=True)

    mnist_classifier.train(
        input_fn=train_input_fn)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)

    accuracy = eval_results["accuracy"] * 100
    # Set the best acc if we have a new best or if it is the first step
    if accuracy > best_acc or i == 0:
        best_acc = accuracy
        print ("=> New Best Accuracy {}".format(accuracy))
    else:
        print("=> Validation Accuracy did not improve")
