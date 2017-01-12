"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/

TODO: Save png to verify input data.
TODO: Scale down the image by 2x.
TODO: Sigmoid vs ReLu.
TODO: Separated target Q network.
"""

from __future__ import print_function
from collections import defaultdict
from collections import deque
import os
import random
import time
import math
import socket
import numpy as np
import tensorflow as tf


RAW_IMAGE = False
if RAW_IMAGE:
  WIDTH = 256
  HEIGHT = 240
else:
  # galaxian.x, missile.y, still enemy xs, 6 incoming enemies and 6 bullets.
  # TODO: Try adding if the galaxian is aimming at a gap.
  INPUT_DIM = 1 + 1 + 10 + 2 * (6 + 6)
  # INPUT_DIM = 1 + 1 + 2 * (1 + 2)

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


class Frame:
  def __init__(self, line):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0

    self.seq = self.NextInt()

    # Cap reward in [-1, 1].
    self.reward = max(-1, min(1, self.NextInt()))

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    galaxian = self.NextPoint()

    missile = self.NextPoint()

    # still enemies (encoded)
    dx = self.NextInt()
    masks = []
    for i in xrange(10):
      masks.append(self.NextInt())

    incoming_enemies = []
    for i in xrange(self.NextInt()):
      incoming_enemies.append(self.NextPoint())

    bullets = []
    for i in xrange(self.NextInt()):
      bullets.append(self.NextPoint())

    if not RAW_IMAGE:
      data = []

      data.append((galaxian.x - 128) / 128.0)

      if missile.y < 200:
        data.append(missile.y / 200.0)
      else:
        data.append(0)

      # still_enemies
      still_enemy_xs = []
      for mask in masks:
        x = (dx + 48 + 8 + 16 * i) % 256;
        if mask:
          still_enemy_xs.append(Frame.InvertX(x - galaxian.x))
      still_enemy_xs.sort(key = abs, reverse = True)
      for x in still_enemy_xs:
        data.append(x)
      for i in xrange(10 - len(still_enemy_xs)):
        data.append(0)

      for e in incoming_enemies:
        dx = Frame.InvertX(e.x - galaxian.x)
        dy = e.y / 200.0
        data.append(dx)
        data.append(dy)
      for i in xrange(6 - len(incoming_enemies)):
        data.append(0)
        data.append(0)

      for e in bullets:
        dx = Frame.InvertX(e.x - galaxian.x)
        dy = e.y / 200.0
        data.append(dx)
        data.append(dy)
      for i in xrange(6 - len(bullets)):
        data.append(0)
        data.append(0)

      assert len(data) == INPUT_DIM
      self.data = np.array(data)
    else:
      self.data = np.zeros((WIDTH, HEIGHT))

      self.AddRect(galaxian, 16, 16)

      if missile.y < 200:
        self.AddRect(missile, 4, 12)

      still_enemies = []
      for mask in masks:
        x = (dx + 48 + 8 + 16 * i) % 256;
        y = 108
        while mask:
          if mask % 2:
            still_enemies.append(Point(x, y))
          mask /= 2
          y -= 12
      assert len(still_enemies) <= 46
      for e in still_enemies:
        self.AddRect(e, 10, 12)

      for e in incoming_enemies:
        self.AddRect(e, 10, 12)

      for b in bullets:
        self.AddRect(b, 4, 12)

  @staticmethod
  def InvertX(dx):
    if dx > 0:
      return (256 - dx) / 256.0
    elif dx < 0:
      return (-256 - dx) / 256.0
    else:
      return 1

  def AddSnapshotsFromSelf(self):
    if not RAW_IMAGE:
      self.datax = np.reshape(self.data, (INPUT_DIM, 1))
      for i in xrange(NUM_SNAPSHOTS-1):
        self.datax = np.append(
            self.datax,
            np.reshape(self.data, (INPUT_DIM, 1)),
            axis = 1)
    else:
      self.datax = np.reshape(self.data, (WIDTH, HEIGHT, 1))
      for i in xrange(NUM_SNAPSHOTS-1):
        self.datax = np.append(
            self.datax,
            np.reshape(self.data, (WIDTH, HEIGHT, 1)),
            axis = 2)

  def AddSnapshotsFromPrev(self, prev_frame):
    if not RAW_IMAGE:
      self.datax = np.append(
          np.reshape(self.data, (INPUT_DIM, 1)),
          prev_frame.datax[:, :NUM_SNAPSHOTS-1],
          axis = 1)
    else:
      self.datax = np.append(
          np.reshape(self.data, (WIDTH, HEIGHT, 1)),
          prev_frame.datax[:, :, :NUM_SNAPSHOTS-1],
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
    self.data[x1:x2, y1:y2] += np.full((x2-x1, y2-y1,), 1.)

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


class NeuralNetwork:
  def __init__(self):
    var = lambda shape: tf.Variable(tf.random_normal(shape))

    if not RAW_IMAGE:
      # Input.
      self.input = tf.placeholder(tf.float32,
          [None, INPUT_DIM, NUM_SNAPSHOTS])
      print('input:', self.input.get_shape())

      # Flatten input.
      INPUT_FLAT_DIM = INPUT_DIM * NUM_SNAPSHOTS
      input_flat = tf.reshape(self.input, [-1, INPUT_FLAT_DIM])

      # Full connected 1.
      self.w1 = var([INPUT_FLAT_DIM, 16])
      self.b1 = var([16])
      fc1 = tf.nn.relu(tf.matmul(input_flat, self.w1) + self.b1)

      # Full connected 2.
      self.w2 = var([16, 8])
      self.b2 = var([8])
      fc2 = tf.nn.relu(tf.matmul(fc1, self.w2) + self.b2)

      # Output.
      self.w3 = var([8, OUTPUT_DIM])
      self.b3 = var([OUTPUT_DIM])
      self.output_layer = (tf.matmul(fc2, self.w3) + self.b3)
    else:
      # Input image.
      self.input = tf.placeholder(tf.float32,
          [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])
      print('input:', self.input.get_shape())

      # Conv 1: 16x12 is the enemy size.
      conv1 = tf.nn.relu(tf.nn.conv2d(
        self.input, var([32, 24, 5, 64]), strides = [1, 16, 12, 1],
        padding = "VALID")
        + var([64]))
      print('conv1:', conv1.get_shape())

      # Conv 2.
      conv2 = tf.nn.relu(tf.nn.conv2d(
        conv1, var([4, 4, 64, 128]), strides = [1, 2, 2, 1],
        padding = "VALID")
        + var([128]))
      print('conv2:', conv2.get_shape())

      # Conv 3.
      conv3 = tf.nn.relu(tf.nn.conv2d(
        conv2, var([2, 2, 128, 128]), strides = [1, 1, 1, 1],
        padding = "VALID")
        + var([128]))
      print('conv3:', conv3.get_shape())

      # Flatten conv 3.
      conv3_flat = tf.reshape(conv3, [-1, 4480])

      # Full connected 4.
      fc4 = tf.nn.relu(tf.matmul(conv3_flat, var([4480, 784])) + var([784]))

      # Output.
      self.output_layer = (tf.matmul(fc4, var([784, OUTPUT_DIM])) +
          var([OUTPUT_DIM]))

    # Training.
    # Error clipping to [-1, 1]?
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
        frame1_batch[i].reward if frame1_batch[i].reward <= 0 else
        frame1_batch[i].reward + GAMMA * np.max(q1_batch[i])
        for i in xrange(len(mini_batch))
    ]

    feed_dict = {
        self.input: [f.datax for f in frame_batch],
        self.action: action_batch,
        self.q_target: q_target_batch,
    }
    self.optimizer.run(feed_dict = feed_dict)
    return self.cost.eval(feed_dict = feed_dict), q_target_batch[-1]

  def Std(self):
    return (
        np.std(self.w1.eval()),
        np.std(self.b1.eval()),
        np.std(self.w2.eval()),
        np.std(self.b2.eval()),
        np.std(self.w3.eval()),
        np.std(self.b3.eval()),
        )


# Hyperparameters.
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE_STEPS = 500000
OBSERVE_STEPS = 0 # 10000
REPLAY_MEMORY = 10000 # 2000  # ~6G memory
MINI_BATCH_SIZE = 32
TRAIN_INTERVAL = 1

CHECKPOINT_DIR = 'galaxian2b/'
CHECKPOINT_FILE = 'model.ckpt'
SAVE_INTERVAL = 1000 # 10000


def FormatList(l):
  return '[' + ' '.join(['%7.3f' % x for x in l]) + ']'

def Run():
  memory = deque()
  nn = NeuralNetwork()
  game = Game()
  frame = game.Step('_')
  frame.AddSnapshotsFromSelf()

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restored from", ckpt.model_checkpoint_path)
    else:
      print("No checkpoint found")

    steps = 0
    epsilon = INITIAL_EPSILON
    cost = 1e9
    q_target_val = -1
    while True:
      if random.random() <= epsilon:
        q_val = []
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
        cost, q_target_val = nn.Train(mini_batch)

      frame = frame1
      steps += 1
      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if steps % SAVE_INTERVAL == 0:
        save_path = saver.save(sess, CHECKPOINT_DIR + CHECKPOINT_FILE,
                               global_step = steps)
        print("Saved to", save_path)

      print("Step %d epsilon: %.6f nn: %s q: %-33s action: %s reward: %2.0f "
          "cost: %8.3f q_target: %8.3f" %
          (steps, epsilon, FormatList(nn.Std()), FormatList(q_val),
            frame1.action, frame1.reward, cost, q_target_val))


if __name__ == '__main__':
  Run()
