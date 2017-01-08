"""Galaxian NN trainer.

Read data on human playing Galaxian, train a Neural Network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import defaultdict

import numpy
import tensorflow as tf
import random

from google3.pyglib import app
from google3.pyglib import flags

FLAGS = flags.FLAGS


SIGHT_X = 48
SIGHT_Y = 32
NUM_SNAPSHOTS = 3


def GenerateInputMapping():
  mapping = {}

  def Add(i):
    mapping[i] = len(mapping)

  for gid in xrange(NUM_SNAPSHOTS):
    for ix in xrange(-SIGHT_X, SIGHT_X):
      Add("%s%d.%03d" % ("s", gid, ix))
  for t in ['i', 'b']:
    for gid in xrange(NUM_SNAPSHOTS):
      for ix in xrange(-SIGHT_X, SIGHT_X):
        for iy in xrange(SIGHT_Y):
          Add("%s%d.%03d,%03d" % (t, gid, ix, iy))
  Add("m")
  Add("gx")
  return mapping


INPUT_MAPPING = GenerateInputMapping()
OUTPUT_MAPPING = {
    'left': 0,
    'right': 1,
    'A': 2,
}

INPUT_DIM = len(INPUT_MAPPING)
OUTPUT_DIM = len(OUTPUT_MAPPING)  # TODO(kelvinlau): 2?
H1_DIM = 20
H2_DIM = 10


def ToMatrix(frames):
  xs = []
  ys = []
  for inputs, outputs in frames:
    x = [0] * len(INPUT_MAPPING)
    y = [0] * 3
    for i, v in inputs.iteritems():
      x[INPUT_MAPPING[i]] = v
    for btn in outputs:
      y[OUTPUT_MAPPING[btn]] = 1
    xs.append(x)
    ys.append(y)
  return numpy.array(xs), numpy.array(ys)


def ReadFrames(f):
  inputs = {}
  outputs = None
  key = None
  for line in f:
    line = line.strip()
    if line == 'done':
      assert key is None
      if outputs is not None:
        yield (inputs, outputs)
        inputs = {}
        outputs = None
      else:
        outputs = set()
    elif key:
      assert inputs is not None
      inputs[key] = line
      key = None
    else:
      if outputs is not None:
        outputs.add(line)
      else:
        key = line


def GetControls(y):
  l = tf.slice(y, [0, 0], [-1, 1])
  r = tf.slice(y, [0, 1], [-1, 1])
  a = tf.slice(y, [0, 2], [-1, 1])
  d = tf.sign(r - l) * tf.sign(tf.floor(tf.abs(r - l) * 10))
  aa = tf.sign(tf.maximum(a - 0.1, 0))
  return tf.concat(1, [d, aa])


def main(unused_argv):
  print("Input dimensions:", len(INPUT_MAPPING))

  frames = []
  with open('galaxian.in', 'r') as f:
    for frame in ReadFrames(f):
      frames.append(frame)

  print("Read %d frames" % len(frames))
  print("Random 10 frames:")
  for j in xrange(10):
    inputs, outputs = frames[random.randint(0, len(frames)-1)]
    print("Input size =", len(inputs), "Output:", outputs)
  output_summary = defaultdict(int)
  for _, outputs in frames:
    output_summary[','.join(outputs)] += 1
  print("Output summary:")
  for buttons, freq in output_summary.items():
    print(1.0*freq/len(frames), freq, buttons)

  train_size = int(len(frames) * 0.8)
  test_size = len(frames) - train_size
  train_x, train_y = ToMatrix(frames[:train_size])
  test_x, test_y = ToMatrix(frames[train_size:])
  print("Train set shapes: x:", train_x.shape, "y:", train_y.shape)
  print("Test set shapes: x:", test_x.shape, "y:", test_y.shape)

  sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, INPUT_DIM])
  y_ = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])

  w1 = tf.Variable(tf.truncated_normal((INPUT_DIM, H1_DIM), stddev=0.1))
  b1 = tf.Variable(tf.constant(0.1, shape=(H1_DIM,)))
  h1 = tf.nn.relu(tf.matmul(x, w1) + b1)

  w2 = tf.Variable(tf.truncated_normal((H1_DIM, H2_DIM), stddev=0.1))
  b2 = tf.Variable(tf.constant(0.1, shape=(H2_DIM,)))
  h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)

  keep_prob = tf.placeholder(tf.float32)
  h2_drop = tf.nn.dropout(h2, keep_prob)

  w3 = tf.Variable(tf.truncated_normal((H2_DIM, OUTPUT_DIM), stddev=0.1))
  b3 = tf.Variable(tf.constant(0.1, shape=(OUTPUT_DIM,)))
  y = tf.matmul(h2_drop, w3) + b3

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  truth = GetControls(y_)
  prediction = GetControls(y)
  correct_prediction = tf.reduce_all(tf.equal(truth, prediction), 1)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess.run(tf.global_variables_initializer())
  batch_size = 50
  for i in xrange(0, train_size, batch_size):
    batch_x = train_x[i:i+batch_size]
    batch_y = train_y[i:i+batch_size]
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch_x, y_: batch_y, keep_prob: 1.0})
      print("Step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

  print("Test accuracy %g"%accuracy.eval(feed_dict={
      x: test_x, y_: test_y, keep_prob: 1.0}))
  for i in xrange(20):
    feed_dict = {
        x: test_x[i: i+1],
        y_: test_y[i: i+1],
        keep_prob: 1.0,
    }
    print("Test", i, ":",
          "y_:", test_y[i],
          "y:", y.eval(feed_dict=feed_dict),
          "truth:", truth.eval(feed_dict=feed_dict),
          "prediction:", prediction.eval(feed_dict=feed_dict),
          "correct:", correct_prediction.eval(feed_dict=feed_dict))

  prediction_summary = defaultdict(int)
  for out in prediction.eval(feed_dict={x: test_x, keep_prob: 1.0}):
    assert len(out) == 2
    buttons = []
    if out[0] < 0:
      buttons.append("left")
    elif out[0] > 0:
      buttons.append("right")
    if out[1]:
      buttons.append("A")
    prediction_summary[','.join(buttons)] += 1
  print("Prediction summary:")
  for buttons, freq in prediction_summary.items():
    print(1.0*freq/test_size, freq, buttons)


if __name__ == '__main__':
  app.run()
