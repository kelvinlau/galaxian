"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
Double Q Learning: https://arxiv.org/pdf/1509.06461v3.pdf

TODO: Save png to verify input data.
TODO: Scale down the image by 2x.
TODO: Random no-op actions at the start of episodes.
TODO: Training the model only on score increases.
"""

from __future__ import print_function
from collections import defaultdict
from collections import deque
import os
import random
import time
import math
import socket
import cv2
import numpy as np
import tensorflow as tf


# Game input/output.
NUM_STILL_ENEMIES = 10
NUM_INCOMING_ENEMIES = 6
NUM_BULLETS = 6

RAW_IMAGE = False
if RAW_IMAGE:
  SCALE = 2
  WIDTH = 256/SCALE
  HEIGHT = 240/SCALE
  SIDE = 84
  NUM_SNAPSHOTS = 4
else:
  DX = 8
  WIDTH = 256 / DX
  INPUT_DIM = 1 + 1 + 5 + 4 + 2 * WIDTH


ACTION_NAMES = ['_', 'L', 'R', 'A']
ACTION_ID = {ACTION_NAMES[i]: i for i in xrange(len(ACTION_NAMES))}
OUTPUT_DIM = len(ACTION_NAMES)  # TODO(kelvinlau): 6?

# Hyperparameters.
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORE_STEPS = 1000000
OBSERVE_STEPS = 5000
REPLAY_MEMORY = 100000 if not RAW_IMAGE else 2000  # 2000 = ~6G memory
MINI_BATCH_SIZE = 32
TRAIN_INTERVAL = 1
UPDATE_TARGET_NETWORK_INTERVAL = 10000

DOUBLE_Q = True
if DOUBLE_Q:
  FINAL_EPSILON = 0.01

# Checkpoint.
CHECKPOINT_DIR = 'galaxian2n/'
CHECKPOINT_FILE = 'model.ckpt'
SAVE_INTERVAL = 10000


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


def OneHot(n, i):
  return [1 if j == i else 0 for j in xrange(n)]


def NumBits(mask):
  ret = 0
  while mask:
    if mask % 2:
      ret += 1
    mask = mask / 2
  return ret


def Sign(x):
  if x == 0:
    return 0
  return 1 if x > 0 else -1


class Frame:
  def __init__(self, line):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0

    self.seq = self.NextInt()

    # Cap reward in [-1, +1].
    self.reward = max(-1, min(1, self.NextInt()))

    self.terminal = self.NextInt()

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    galaxian = self.NextPoint()
    self.galaxian = galaxian

    self.missile = self.NextPoint()

    # still enemies (encoded)
    self.sdx = self.NextInt()
    self.masks = []
    for i in xrange(10):
      self.masks.append(self.NextInt())
    self.masks = self.masks[:NUM_STILL_ENEMIES]

    self.incoming_enemies = {}
    for i in xrange(self.NextInt()):
      eid = self.NextInt()
      self.incoming_enemies[eid] = self.NextPoint()

    self.bullets = {}
    for i in xrange(self.NextInt()):
      bid = self.NextInt()
      self.bullets[bid] = self.NextPoint()

    if RAW_IMAGE:
      self.data = np.zeros((WIDTH, HEIGHT))

      self.AddRect(galaxian, 16, 16, .5)

      if self.missile.y < 200:
        self.AddRect(self.missile, 4, 8, .5)

      still_enemies = []
      for mask in self.masks:
        x = self.sdx + 16 * i
        y = 108
        while mask:
          if mask % 2:
            still_enemies.append(Point(x, y))
          mask /= 2
          y -= 12
      assert len(still_enemies) <= 46
      for e in still_enemies:
        self.AddRect(e, 8, 12)

      for e in self.incoming_enemies.values():
        self.AddRect(e, 8, 12)

      for b in self.bullets.values():
        self.AddRect(b, 4, 12)

      self.data = cv2.resize(self.data, (SIDE, SIDE))

  @staticmethod
  def InvertX(dx):
    if dx > 0:
      return (256 - dx) / 256.0
    elif dx < 0:
      return (-256 - dx) / 256.0
    else:
      return 1

  # TODO: call this in Game.Step()
  def AddPrev(self, prev_frames):
    if not RAW_IMAGE:
      self.data = []

      # missile y
      if self.missile.y < 200:
        self.data.append(self.missile.y / 200.0)
      else:
        self.data.append(0)

      # galaxian x
      galaxian = self.galaxian
      self.data.append((galaxian.x - 128) / 128.0)

      # closest still enemies dx relative to galaxian
      # one for left, one for right, and vx.
      sl = (-1000, 0)
      sr = (+1000, 0)
      for i in xrange(len(self.masks)):
        mask = self.masks[i]
        if mask:
          x = self.sdx + 16 * i
          num = NumBits(mask)
          if x <= galaxian.x and x > sl[0]:
            sl = (x, num)
          if x >= galaxian.x and x < sr[0]:
            sr = (x, num)
      svx = 0
      for f in reversed(prev_frames):
        if f.sdx != self.sdx:
          svx = Sign(self.sdx - f.sdx)
          break
      self.data += [min(galaxian.x-sl[0],32)/32., sl[1] / 7.]
      self.data += [min(sr[0]-galaxian.x,32)/32., sr[1] / 7.]
      self.data.append(svx)

      # closest incoming enemy x, y relative to galaxian, and vx, vy
      ci = []
      if self.incoming_enemies:
        eid, e = max(self.incoming_enemies.items(), key = lambda p: p[1].y)
        dx = (e.x - galaxian.x) / 256.0
        dy = (e.y - galaxian.y) / 200.0
        ci += [dx, dy]
        if prev_frames:
          prev = prev_frames[0]
          if eid in prev.incoming_enemies:
            pe = prev.incoming_enemies[eid]
            dx = (e.x - pe.x) / 256.0
            dy = (e.y - pe.y) / 200.0
            ci += [dx, dy]
        if len(ci) == 2:
          ci += [0, 0]
      else:
        ci = [3, 3, 3, 3]
      self.data += ci

      # hit map
      hmap = [0.] * (WIDTH * 2)
      if prev_frames:
        prev = prev_frames[0]  # TODO: use furthest frame with same enemies/bullets
        steps = len(prev_frames)
        y = 206  # galaxian top y
        for eid, e in self.incoming_enemies.iteritems():
          if eid in prev.incoming_enemies:
            pe = prev.incoming_enemies[eid]
            if pe.y < e.y < y:
              x = (e.x-pe.x)*1.0/(e.y-pe.y)*(y-pe.y)+pe.x
              t = (y-e.y)*1.0/(e.y-pe.y)*steps
              hmap[max(0, min(2*WIDTH-1, int(round(x-galaxian.x+256))/DX))] += max(0., 1.-t/24.)
        for eid, e in self.bullets.iteritems():
          if eid in prev.bullets:
            pe = prev.bullets[eid]
            if pe.y < e.y < y:
              x = (e.x-pe.x)*1.0/(e.y-pe.y)*(y-pe.y)+pe.x
              t = (y-e.y)*1.0/(e.y-pe.y)*steps
              hmap[max(0, min(2*WIDTH-1, int(round(x-galaxian.x+256))/DX))] += max(0., 1.-t/12.)
      self.data += hmap

      if not self.terminal:
        ix = galaxian.x / DX
        self.reward -= min(1., hmap[ix]) * .5

      assert len(self.data) == INPUT_DIM
      self.datax = np.array(self.data)
    else:
      if not prev_frames:
        self.datax = np.reshape(self.data, (SIDE, SIDE, 1))
        for i in xrange(NUM_SNAPSHOTS-1):
          self.datax = np.append(
              self.datax,
              np.reshape(self.data, (SIDE, SIDE, 1)),
              axis = 2)
      else:
        prev_frame = prev_frames[-1]
        self.datax = np.append(
            np.reshape(self.data, (SIDE, SIDE, 1)),
            prev_frame.datax[:, :, :NUM_SNAPSHOTS-1],
            axis = 2)

  def NextToken(self):
    self._idx += 1
    return self._tokens[self._idx - 1]

  def NextInt(self):
    return int(self.NextToken())

  def NextPoint(self):
    return Point(self.NextInt(), self.NextInt())

  def AddRect(self, c, w, h, v=1.):
    c.x /= SCALE
    c.y /= SCALE
    w /= 2 * SCALE
    h /= 2 * SCALE
    x1 = max(c.x - w, 0)
    x2 = min(c.x + w, WIDTH)
    y1 = max(c.y - h, 0)
    y2 = min(c.y + h, HEIGHT)
    if x1 >= x2 or y1 >= y2:
      return
    self.data[x1:x2, y1:y2] += np.full((x2-x1, y2-y1,), v)

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


def ClippedError(x):
  # Huber loss
  return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class NeuralNetwork:
  def __init__(self, name, trainable=True):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02), trainable=trainable)

    with tf.variable_scope(name):
      if not RAW_IMAGE:
        # Input.
        self.input = tf.placeholder(tf.float32, [None, INPUT_DIM])
        print('input:', self.input.get_shape())

        # Fully connected 1.
        self.w1 = var([INPUT_DIM, 16])
        self.b1 = var([16])
        fc1 = tf.nn.relu(tf.matmul(self.input, self.w1) + self.b1)

        # Fully connected 2.
        self.w2 = var([16, 8])
        self.b2 = var([8])
        fc2 = tf.nn.relu(tf.matmul(fc1, self.w2) + self.b2)

        # Output.
        self.w3 = var([8, OUTPUT_DIM])
        self.b3 = var([OUTPUT_DIM])
        self.output = (tf.matmul(fc2, self.w3) + self.b3)
      else:
        # Input image.
        self.input = tf.placeholder(tf.float32,
            [None, SIDE, SIDE, NUM_SNAPSHOTS])
        print('input:', self.input.get_shape())

        # Conv 1.
        self.w1 = var([8, 8, NUM_SNAPSHOTS, 32])
        self.b1 = var([32])
        conv1 = tf.nn.relu(tf.nn.conv2d(
          self.input, self.w1, strides = [1, 4, 4, 1], padding = "VALID")
          + self.b1)
        print('conv1:', conv1.get_shape())

        # Conv 2.
        self.w2 = var([4, 4, 32, 64])
        self.b2 = var([64])
        conv2 = tf.nn.relu(tf.nn.conv2d(
          conv1, self.w2, strides = [1, 2, 2, 1], padding = "VALID")
          + self.b2)
        print('conv2:', conv2.get_shape())

        # Conv 3.
        self.w3 = var([3, 3, 64, 64])
        self.b3 = var([64])
        conv3 = tf.nn.relu(tf.nn.conv2d(
          conv2, self.w3, strides = [1, 1, 1, 1], padding = "VALID")
          + self.b3)
        print('conv3:', conv3.get_shape())

        # Flatten conv 3.
        conv3_flat = tf.reshape(conv3, [-1, 3136])

        # Fully connected 4.
        self.w4 = var([3136, 512])
        self.b4 = var([512])
        fc4 = tf.nn.relu(tf.matmul(conv3_flat, self.w4) + self.b4)

        # Output.
        self.w5 = var([512, OUTPUT_DIM])
        self.b5 = var([OUTPUT_DIM])
        self.output = (tf.matmul(fc4, self.w5) + self.b5)

    self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    assert len(self.theta) == (10 if RAW_IMAGE else 6), len(self.theta)

    if trainable:
      # Training.
      self.action = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
      self.y = tf.placeholder(tf.float32, [None])
      q_action = tf.reduce_sum(tf.mul(self.output, self.action),
          reduction_indices = 1)
      self.cost = tf.reduce_mean(ClippedError(q_action - self.y))
      self.optimizer = tf.train.RMSPropOptimizer(
          learning_rate=0.00025, momentum=.95, epsilon=1e-2).minimize(self.cost)

  def Vars(self):
    return self.theta

  def Eval(self, frames):
    return self.output.eval(feed_dict = {
        self.input: [f.datax for f in frames]
    })

  def Train(self, tnn, mini_batch):
    frame_batch = [d[0] for d in mini_batch]
    action_batch = [d[1] for d in mini_batch]
    frame1_batch = [d[2] for d in mini_batch]

    t_q1_batch = tnn.Eval(frame1_batch)
    y_batch = [0] * len(mini_batch)
    if not DOUBLE_Q:
      for i in xrange(len(mini_batch)):
        reward = frame1_batch[i].reward
        if frame1_batch[i].terminal:
          y_batch[i] = reward
        else:
          y_batch[i] = reward + GAMMA * np.max(t_q1_batch[i])
    else:
      q1_batch = self.Eval(frame1_batch)
      for i in xrange(len(mini_batch)):
        reward = frame1_batch[i].reward
        if frame1_batch[i].terminal:
          y_batch[i] = reward
        else:
          y_batch[i] = reward + GAMMA * t_q1_batch[i][np.argmax(q1_batch[i])]

    feed_dict = {
        self.input: [f.datax for f in frame_batch],
        self.action: action_batch,
        self.y: y_batch,
    }
    self.optimizer.run(feed_dict = feed_dict)
    return self.cost.eval(feed_dict = feed_dict), y_batch[-1]

  def CopyFrom(self, sess, src):
    for v1, v2 in zip(self.Vars(), src.Vars()):
      sess.run(v1.assign(v2))

  def CheckSum(self):
    return [np.sum(var.eval()) for var in self.Vars()]


def FormatList(l):
  return '[' + ' '.join(['%7.3f' % x for x in l]) + ']'

def Run():
  memory = deque()
  memoryx = deque()
  nn = NeuralNetwork('nn')
  tnn = NeuralNetwork('tnn', trainable=False)
  game = Game()
  prev_frames = deque()
  frame = game.Step('_')
  frame.AddPrev(prev_frames)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(nn.Vars())
    if not os.path.exists(CHECKPOINT_DIR):
      os.makedirs(CHECKPOINT_DIR)
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restored from", ckpt.model_checkpoint_path)
    else:
      print("No checkpoint found")

    tnn.CopyFrom(sess, nn)

    steps = 0
    epsilon = INITIAL_EPSILON
    cost = 1e9
    y_val = 1e9
    while True:
      if random.random() <= epsilon:
        q_val = []
        action = ACTION_NAMES[random.randrange(OUTPUT_DIM)]
      else:
        q_val = nn.Eval([frame])[0]
        action = ACTION_NAMES[np.argmax(q_val)]

      frame1 = game.Step(action)
      frame1.AddPrev(prev_frames)

      action_val = np.zeros([OUTPUT_DIM], dtype=np.int)
      action_val[frame1.action_id] = 1

      # TODO: no need to store action_val in memory, it's in frame1 already.
      memory.append((frame, action_val, frame1))
      if len(memory) > REPLAY_MEMORY:
        memory.popleft()
      if frame1.reward != 0:
        memoryx.append((frame, action_val, frame1))
        if len(memoryx) > REPLAY_MEMORY:
          memoryx.popleft()

      prev_frames.append(frame1)
      if len(prev_frames) > 4:
        prev_frames.popleft()

      if steps % TRAIN_INTERVAL == 0 and steps > OBSERVE_STEPS:
        mini_batch = random.sample(memory, min(len(memory), MINI_BATCH_SIZE))
        mini_batch += random.sample(memoryx, min(len(memoryx), MINI_BATCH_SIZE))
        mini_batch.append(memory[-1])
        cost, y_val = nn.Train(tnn, mini_batch)

      frame = frame1
      steps += 1

      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if steps % UPDATE_TARGET_NETWORK_INTERVAL == 0:
        print('Target network before:', tnn.CheckSum())
        tnn.CopyFrom(sess, nn)
        print('Target network after:', tnn.CheckSum())

      if steps % SAVE_INTERVAL == 0:
        save_path = saver.save(sess, CHECKPOINT_DIR + CHECKPOINT_FILE,
                               global_step = steps)
        print("Saved to", save_path)

      print("Step %d epsilon: %.6f nn: %s q: %-33s action: %s reward: %5.2f "
          "cost: %8.3f y: %8.3f" %
          (steps, epsilon, FormatList(nn.CheckSum()), FormatList(q_val),
            frame1.action, frame1.reward, cost, y_val))


if __name__ == '__main__':
  Run()
