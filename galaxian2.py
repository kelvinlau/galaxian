"""Galaxian NN trainer.

Galaxian deep neural network.

Ref: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

TODO: Save png to verify input data.
TODO: Scale down the image by 2x.
"""

from __future__ import print_function
from collections import defaultdict
from collections import deque
import os
import random
import time
import socket
import numpy as np
import tensorflow as tf


WIDTH = 256
HEIGHT = 240
NUM_SNAPSHOTS = 5
ACTION_NAMES = ['_', 'L', 'R', 'A']
ACTION_ID = {'_': 0, 'L': 1, 'R': 2, 'A': 3}
OUTPUT_DIM = len(ACTION_NAMES)  # TODO(kelvinlau): 6?


class Timer:
  def __init__(self, name):
    self.name = name

  def __enter__(self):
    self.start = int(time.time() * 1e6)
    return self

  def __exit__(self, *args):
    self.end = int(time.time() * 1e6)
    self.interval = self.end - self.start
    print(self.name, self.interval, self.start, self.end)


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y


class Frame2:
  def __init__(self, line):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0

    self.seq = self.NextInt()

    self.reward = self.NextInt()

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    self.image = np.zeros((WIDTH, HEIGHT))

    galaxian = self.NextPoint()
    self.AddRect(galaxian, 16, 16)

    missile = self.NextPoint()
    if missile.y < 200:
      self.AddRect(missile, 4, 12)

    still_enemies = []
    dx = self.NextInt()
    for i in xrange(10):
      x = (dx + 48 + 8 + 16 * i) % 256;
      y = 108
      mask = self.NextInt()
      while mask:
        if mask % 2:
          still_enemies.append(Point(x, y))
        mask /= 2
        y -= 12
    assert len(still_enemies) <= 46
    for e in still_enemies:
      self.AddRect(e, 10, 12)

    incoming_enemies = []
    for i in xrange(self.NextInt()):
      incoming_enemies.append(self.NextPoint())
    for e in incoming_enemies:
      self.AddRect(e, 10, 12)

    bullets = []
    for i in xrange(self.NextInt()):
      bullets.append(self.NextPoint())
    for b in bullets:
      self.AddRect(b, 4, 12)

  def AddSnapshotsFromSelf(self):
    self.imagex = np.reshape(self.image, (WIDTH, HEIGHT, 1))
    for i in xrange(NUM_SNAPSHOTS-1):
      self.imagex = np.append(
          self.imagex,
          np.reshape(self.image, (WIDTH, HEIGHT, 1)),
          axis = 2)

  def AddSnapshotsFromPrev(self, prev_frame):
    self.imagex = np.append(
        np.reshape(self.image, (WIDTH, HEIGHT, 1)),
        prev_frame.imagex[:, :, :NUM_SNAPSHOTS-1],
        axis = 2)

  def NextToken(self):
    self._idx += 1
    return self._tokens[self._idx - 1]

  def NextInt(self):
    return int(self.NextToken())

  def NextPoint(self):
    return Point(self.NextInt(), self.NextInt())

  def AddRect(self, c, w, h):
    w /= 2
    h /= 2
    x1 = max(c.x - w, 0)
    x2 = min(c.x + w, WIDTH)
    y1 = max(c.y - h, 0)
    y2 = min(c.y + h, HEIGHT)
    if x1 >= x2 or y1 >= y2:
      return
    self.image[x1:x2, y1:y2] += np.full((x2-x1, y2-y1,), 1.)


class Frame:
  # galaxian.x, missile.y, still enemies dx and 10 masks, 6 incoming enemies and
  # 6 bullets.
  # TODO: Try removing the 11.
  # TODO: Try adding if the galaxian is aimming at a gap.
  INPUT_DIM = 1 + 1 + 11 + 2 * (6 + 6)

  def __init__(self, line):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0

    self.seq = self.NextInt()

    self.reward = max(0, self.NextInt() + 1)

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    data = []

    galaxian = self.NextPoint()
    data.append((galaxian.x - 128) / 128.0)

    missile = self.NextPoint()
    if missile.y < 200:
      data.append(missile.y / 200.0)
    else:
      data.append(0)

    # still_enemies
    dx = self.NextInt()
    data.append((dx - galaxian.x) / 256.0)
    for i in xrange(10):
      mask = self.NextInt()
      data.append(1 if mask else 0)

    num_incoming_enemies = self.NextInt()
    for i in xrange(num_incoming_enemies):
      e = self.NextPoint()
      dx = Frame.InvertX(e.x - galaxian.x)
      dy = e.y / 200.0
      data.append(dx)
      data.append(dy)
    for i in xrange(6 - num_incoming_enemies):
      data.append(0)
      data.append(0)

    num_bullets = self.NextInt()
    for i in xrange(num_bullets):
      e = self.NextPoint()
      dx = Frame.InvertX(e.x - galaxian.x)
      dy = e.y / 200.0
      data.append(dx)
      data.append(dy)
    for i in xrange(6 - num_bullets):
      data.append(0)
      data.append(0)

    assert len(data) == self.INPUT_DIM
    self.data = np.array(data)

  @staticmethod
  def InvertX(dx):
    if dx > 0:
      return (256 - dx) / 256.0
    elif dx < 0:
      return (-256 - dx) / 256.0
    else:
      return 1

  def AddSnapshotsFromSelf(self):
    self.datax = np.reshape(self.data, (self.INPUT_DIM, 1))
    for i in xrange(NUM_SNAPSHOTS-1):
      self.datax = np.append(
          self.datax,
          np.reshape(self.data, (self.INPUT_DIM, 1)),
          axis = 1)

  def AddSnapshotsFromPrev(self, prev_frame):
    self.datax = np.append(
        np.reshape(self.data, (self.INPUT_DIM, 1)),
        prev_frame.datax[:, :NUM_SNAPSHOTS-1],
        axis = 1)

  def NextToken(self):
    self._idx += 1
    return self._tokens[self._idx - 1]

  def NextInt(self):
    return int(self.NextToken())

  def NextPoint(self):
    return Point(self.NextInt(), self.NextInt())

  def CheckSum(self):
    return np.sum(self.datax)


class Game:
  def __init__(self):
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect(('localhost', 62343))
    self._fin = self._sock.makefile()
    self._seq = 0

  def Step(self, action):
    self._seq += 1
    #print()
    #print(action, self._seq)

    self._sock.send(action + ' ' + str(self._seq) + '\n')

    line = self.NextLine()

    frame = Frame(line)

    assert frame.seq == self._seq, 'Expecting %d, got %d' % (self._seq,
        frame.seq)
    # assert frame.action == action, 'Expecting %s, got %s' % (action,
    #     frame.action)

    return frame

  def NextLine(self):
    return self._fin.readline().strip()


def TestGame():
  game = Game()
  while True:
    for i in xrange(10):
      game.Step('L')
    game.Step('A')
    for i in xrange(10):
      game.Step('_')
    for i in xrange(10):
      game.Step('R')
    game.Step('A')
    for i in xrange(10):
      game.Step('_')


class NeuralNetwork2:
  def __init__(self):
    # Input image.
    self.input = tf.placeholder(tf.float32,
        [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])
    print('input:', self.input.get_shape())

    zeros = lambda shape: tf.Variable(tf.zeros(shape))

    # Conv 1: 16x12 is the enemy size.
    conv1 = tf.nn.relu(tf.nn.conv2d(
      self.input, zeros([32, 24, 5, 64]), strides = [1, 16, 12, 1],
      padding = "VALID")
      + zeros([64]))
    print('conv1:', conv1.get_shape())

    # Conv 2.
    conv2 = tf.nn.relu(tf.nn.conv2d(
      conv1, zeros([4, 4, 64, 128]), strides = [1, 2, 2, 1],
      padding = "VALID")
      + zeros([128]))
    print('conv2:', conv2.get_shape())

    # Conv 3.
    conv3 = tf.nn.relu(tf.nn.conv2d(
      conv2, zeros([2, 2, 128, 128]), strides = [1, 1, 1, 1],
      padding = "VALID")
      + zeros([128]))
    print('conv3:', conv3.get_shape())

    # Flatten conv 3.
    conv3_flat = tf.reshape(conv3, [-1, 4480])

    # Full connected 4.
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, zeros([4480, 784])) + zeros([784]))

    # Output.
    self.output_layer = (tf.matmul(fc4, zeros([784, OUTPUT_DIM])) +
        zeros([OUTPUT_DIM]))

    # Training.
    self.action = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
    self.q_target = tf.placeholder(tf.float32, [None])
    q_action = tf.reduce_sum(tf.mul(self.output_layer, self.action),
        reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(q_action - self.q_target))
    self.optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-2, epsilon=1e-3).minimize(cost)

  def Eval(self, frames):
    return self.output_layer.eval(feed_dict = {
        self.input: [f.imagex for f in frames]
    })

  def Train(self, mini_batch):
    frame_batch = [d[0] for d in mini_batch]
    action_batch = [d[1] for d in mini_batch]
    frame1_batch = [d[2] for d in mini_batch]

    q1_batch = self.Eval(frame1_batch)
    q_target_batch = [
        frame1_batch[i].reward if frame1_batch[i].reward <= 0 else
        frame1_batch[i].reward + GAMMA * np.max(q1_batch[i])
        for i in xrange(len(mini_batch))
    ]

    self.optimizer.run(feed_dict = {
        self.input: [f.imagex for f in frame_batch],
        self.action: action_batch,
        self.q_target: q_target_batch,
    })


class NeuralNetwork:
  def __init__(self):
    # Input.
    self.input = tf.placeholder(tf.float32,
        [None, Frame.INPUT_DIM, NUM_SNAPSHOTS])
    print('input:', self.input.get_shape())

    zeros = lambda shape: tf.Variable(tf.zeros(shape))

    # Flatten input.
    INPUT_FLAT_DIM = Frame.INPUT_DIM * NUM_SNAPSHOTS
    input_flat = tf.reshape(self.input, [-1, INPUT_FLAT_DIM])

    # Full connected 1.
    fc1 = tf.nn.relu(tf.matmul(input_flat, zeros([INPUT_FLAT_DIM, 128])) +
        zeros([128]))

    # Full connected 2.
    fc2 = tf.nn.relu(tf.matmul(fc1, zeros([128, 64])) + zeros([64]))

    # Output.
    self.output_layer = (tf.matmul(fc2, zeros([64, OUTPUT_DIM])) +
        zeros([OUTPUT_DIM]))

    # Training.
    self.action = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
    self.q_target = tf.placeholder(tf.float32, [None])
    q_action = tf.reduce_sum(tf.mul(self.output_layer, self.action),
        reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(q_action - self.q_target))
    self.optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-2, epsilon=1e-3).minimize(self.cost)

  def Eval(self, frames):
    return self.output_layer.eval(feed_dict = {
        self.input: [f.datax for f in frames]
    })

  def Train(self, mini_batch):
    frame_batch = [d[0] for d in mini_batch]
    action_batch = [d[1] for d in mini_batch]
    frame1_batch = [d[2] for d in mini_batch]

    q1_batch = self.Eval(frame1_batch)
    q_target_batch = [
        frame1_batch[i].reward if frame1_batch[i].reward < 0 else
        frame1_batch[i].reward + GAMMA * np.max(q1_batch[i])
        for i in xrange(len(mini_batch))
    ]

    feed_dict = {
        self.input: [f.datax for f in frame_batch],
        self.action: action_batch,
        self.q_target: q_target_batch,
    }
    self.optimizer.run(feed_dict = feed_dict)
    return self.cost.eval(feed_dict = feed_dict)


# Deep Learning Params
GAMMA = 0.99
INITIAL_EPSILON = 0.5 # 1.0
FINAL_EPSILON = 0.05 # 0.05
EXPLORE_STEPS = 500000
OBSERVE_STEPS = 0 # 10000
REPLAY_MEMORY = 10000 # 2000  # ~6G memory
MINI_BATCH_SIZE = 100
TRAIN_INTERVAL = 10  # 24

CHECKPOINT_DIR = 'galaxian3b/'
CHECKPOINT_FILE = 'model.ckpt'
SAVE_INTERVAL = 1000 # 10000


def Run():
  memory = deque()
  nn = NeuralNetwork()
  game = Game()
  frame = game.Step('_')
  frame.AddSnapshotsFromSelf()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restored from", ckpt.model_checkpoint_path)
    else:
      print("No checkpoint found")

    steps = 0
    epsilon = INITIAL_EPSILON
    cost = 1e9
    while True:
      if random.random() <= epsilon:
        q_val = None
        action = ACTION_NAMES[random.randrange(OUTPUT_DIM)]
      else:
        q_val = nn.Eval([frame])[0]
        action = ACTION_NAMES[np.argmax(q_val)]

      frame1 = game.Step(action)

      frame1.AddSnapshotsFromPrev(frame)

      action_val = np.zeros([OUTPUT_DIM], dtype=np.int)
      action_val[frame1.action_id] = 1

      memory.append((frame, action_val, frame1))
      if len(memory) > REPLAY_MEMORY:
        memory.popleft()

      if steps % TRAIN_INTERVAL == 0 and steps > OBSERVE_STEPS:
        mini_batch = random.sample(memory, min(len(memory), MINI_BATCH_SIZE))
        mini_batch.append(memory[-1])
        cost = nn.Train(mini_batch)

      qx_val = nn.Eval([frame])[0]

      frame = frame1
      steps += 1
      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if steps % SAVE_INTERVAL == 0:
        save_path = saver.save(sess, CHECKPOINT_DIR + CHECKPOINT_FILE,
                               global_step = steps)
        print("Saved to", save_path)

      print("Step %7d epsilon: %.6f action: %s reward: %4.0f q: %-40s "
          "qx: %-40s cost: %8.2f" %
          (steps, epsilon, frame1.action, frame1.reward,
            ' '.join(map(str, q_val)) if q_val is not None else '',
            ' '.join(map(str, qx_val)) if qx_val is not None else '',
            cost))


if __name__ == '__main__':
  Run()
