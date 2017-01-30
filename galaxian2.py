"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.tithx7juq

TODO: Save png to verify input data.
TODO: Training the model only on score increases?
TODO: 2D sensors.
TODO: In-bound but edge tiles should have some penality?
TODO: Fewer layer.
TODO: Dropout/Bayesian.
TODO: LSTM/A3C.
TODO: tf.nn.elu.
TODO: 2 hmaps.
TODO: 3 smaps.
"""

from __future__ import print_function
from collections import defaultdict
from collections import deque
import os
import random
import time
import math
import socket
import subprocess
import logging
#import cv2
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', '', 'enable logging.debug')
flags.DEFINE_string('server', '', 'server binary')
flags.DEFINE_string('rom', './galaxian.nes', 'galaxian nes rom file')
flags.DEFINE_float('eps', None, 'initial epsilon')
flags.DEFINE_string('checkpoint_dir', 'models/2.27', 'checkpoint dir')
flags.DEFINE_integer('port', 62343, 'server port to conenct')
flags.DEFINE_bool('train_paths', False, 'train pnn')
flags.DEFINE_string('pnn_dir', 'models/pnn10', 'pnn model dir')
flags.DEFINE_bool('send_paths', False, 'send path to render by lua server')


# Game input/output.
NUM_STILL_ENEMIES = 10
NUM_INCOMING_ENEMIES = 7
RAW_IMAGE = False
if RAW_IMAGE:
  NUM_SNAPSHOTS = 4
  SCALE = 2
  WIDTH = 256/SCALE
  HEIGHT = 240/SCALE
  SIDE = 84
else:
  DX = 4
  WIDTH = 256 / DX
  FOCUS = 16
  NUM_SNAPSHOTS = 5
  NIE = 32
  INPUT_DIM = 4 + (2*FOCUS+3) + NIE + WIDTH
PATH_LEN = 12


ACTION_NAMES = ['_', 'L', 'R', 'A', 'l', 'r']
ACTION_ID = {ACTION_NAMES[i]: i for i in xrange(len(ACTION_NAMES))}
OUTPUT_DIM = len(ACTION_NAMES)

# Hyperparameters.
DOUBLE_Q = True
GAMMA = 0.98
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05 if DOUBLE_Q else 0.1
EXPLORE_STEPS = 2000000
OBSERVE_STEPS = 5000
REPLAY_MEMORY = 100000 if not RAW_IMAGE else 2000  # 2000 = ~6G memory
MINI_BATCH_SIZE = 32
TRAIN_INTERVAL = 4
UPDATE_TARGET_NETWORK_INTERVAL = 10000
PATH_TEST_INTERVAL = 1000

# Checkpoint.
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


def one_hot(n, i):
  return [1 if j == i else 0 for j in xrange(n)]


def num_bits(mask):
  ret = 0
  while mask:
    if mask % 2:
      ret += 1
    mask = mask / 2
  return ret


def sign(x):
  if x == 0:
    return 0
  return 1 if x > 0 else -1


def enemy_type(row):
  assert 0 <= row <= 5
  if row <= 2:
    return 0
  elif row == 3:
    return 1
  else:
    return 2


class Frame:
  def __init__(self, line, prev_frames):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0
    logging.debug('%s', self._tokens)

    self.seq = self.NextInt()

    score = self.NextInt()
    self.reward = math.sqrt(score/30.0)

    self.terminal = self.NextInt()

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    # Penalty on holding the A button.
    if prev_frames:
      pv = prev_frames[-1]
      if pv.missile.y < 200 and self.action in ['A', 'l', 'r']:
        self.reward -= 0.25

    galaxian = self.NextPoint()
    self.galaxian = galaxian

    self.missile = self.NextPoint()
    # penalty on miss
    # XXX: this may break if frame skip setting is change (currently 5).
    if self.missile.y <= 4:
      self.reward -= 0.1

    # still enemies (encoded)
    self.sdx = self.NextInt()
    self.masks = []
    for i in xrange(10):
      self.masks.append(self.NextInt())
    self.masks = self.masks[:NUM_STILL_ENEMIES]

    self.incoming_enemies = {}
    for i in xrange(self.NextInt()):
      eid = self.NextInt()
      e = self.NextPoint()
      e.row = self.NextInt()
      self.incoming_enemies[eid] = e

    self.bullets = {}
    for i in xrange(self.NextInt()):
      bid = self.NextInt()
      self.bullets[bid] = self.NextPoint()

    if not RAW_IMAGE:
      self.data = []

      # missile x, y
      self.data.append((self.missile.x - galaxian.x) / 256.0)
      if self.missile.y < 200:
        self.data.append(self.missile.y / 200.0)
      else:
        self.data.append(0)

      # fired or not
      fired = 0
      if prev_frames:
        pv = prev_frames[-1]
        if pv.missile.y >= 200 and pv.action in ['A', 'l', 'r']:
          fired = 1
      self.data.append(fired)
      logging.debug('fired %d missile %d', fired, self.missile.y)

      # galaxian x
      self.data.append((galaxian.x - 128) / 128.0)

      # still enemies dx relative to galaxian in focus region, and vx.
      smap = [0.] * (2*FOCUS)
      x1 = galaxian.x - FOCUS
      x2 = galaxian.x + FOCUS
      sl = 0
      sr = 0
      for i in xrange(len(self.masks)):
        mask = self.masks[i]
        if mask:
          ex = self.sdx + 16 * i
          num = num_bits(mask)
          for x in xrange(max(ex-4, x1), min(ex+4, x2)):
            smap[x-x1] += num / 7.
          if ex < x1:
            sl = 1
          if ex >= x2:
            sr = 1
      svx = 0
      if prev_frames:
        svx = sign(self.sdx - prev_frames[-1].sdx)
      self.data += smap
      self.data.append(sl)
      self.data.append(sr)
      self.data.append(svx)
      logging.debug('smap [%s] %s %s %s',
          ''.join(['x' if h > 0 else '_' for h in smap]), sl, sr, svx)

      # incoming enemy x, y relative to galaxian, and vx, vy
      ies = []
      for eid, e in sorted(
          self.incoming_enemies.items(), key = lambda p: p[1].y):
        dx = (e.x - galaxian.x) / 256.0
        dy = (e.y - galaxian.y) / 200.0
        ie = [dx, dy]
        for pf in reversed(prev_frames):
          if eid not in pf.incoming_enemies:
            break
          pe = pf.incoming_enemies[eid]
          if abs(pe.y - e.y) > 100:
            break
          vx = (e.x - pe.x) / 256.0
          vy = (e.y - pe.y) / 200.0
          ie += [vx, vy]
          e = pe
        assert len(ie) <= 2*NUM_SNAPSHOTS
        ie += [0, 0] * (NUM_SNAPSHOTS - len(ie)/2)
        ies.append(ie)
      for i in xrange(NUM_INCOMING_ENEMIES-len(ies)):
        ies.append([3, 3] + [0, 0] * (NUM_SNAPSHOTS-1))
      self.ies = ies

      # hit map
      def ix(x):
        return max(0, min(WIDTH-1, (x-galaxian.x+128)/DX))
      # out-of-bound tiles have penality.
      hmap = [0. if ix(0) <= i <= ix(255) else 1. for i in range(WIDTH)]
      fmap = [0. if 0 <= i+x1 < 256 else 1. for i in range(FOCUS*2)]
      def fill_fmap(ex, hit):
        for x in range(max(ex-4, x1), min(ex+4, x2)):
          fmap[x-x1] += hit
      if prev_frames:
        steps = len(prev_frames)
        y = galaxian.y
        for eid, e in self.incoming_enemies.iteritems():
          pe = None  # the furthest frame having this enemy
          for pf in prev_frames:
            if eid in pf.incoming_enemies:
              pe = pf.incoming_enemies[eid]
              break
          x, t = None, None
          if pe and pe.y < e.y < y:
            x = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y-pe.y)+pe.x))
            t = (y-e.y)*1.0/(e.y-pe.y)*steps
          elif y <= e.y < y + 12:
            x = e.x
            t = 0
          if x is not None:
            hit = max(0., 1.-t/24.)
            hmap[ix(x)] += hit
            if x1 <= x < x2:
              fill_fmap(x, hit)
        for eid, e in self.bullets.iteritems():
          pe = None  # the furthest frame having this enemy
          for pf in prev_frames:
            if eid in pf.bullets:
              pe = pf.bullets[eid]
              break
          if pe and pe.y < e.y < y:
            x = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y-pe.y)+pe.x))
            t = (y-e.y)*1.0/(e.y-pe.y)*steps
            hit = max(0., 1.-t/12.)
            hmap[ix(x)] += hit
            if x1 <= x < x2:
              fill_fmap(x, hit)
      self.data += hmap
      #self.data += fmap
      logging.debug('hmap [%s]', ''.join(['x' if h>0 else '_' for h in hmap]))
      #logging.debug('fmap [%s]', ''.join(['x' if h>0 else '_' for h in fmap]))

      if not self.terminal:
        self.reward -= min(1., hmap[ix(galaxian.x)]) * .25

      assert len(self.data) == INPUT_DIM-NIE
      self.datax = np.array(self.data)
    else:
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
  def __init__(self, port):
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect(('localhost', port))
    self._fin = self._sock.makefile()
    self._prev_frames = deque()

  def Start(self, seq):
    self._seq = seq
    self._sock.send('galaxian:start %d\n' % (seq+1))
    assert self._fin.readline().strip() == 'ack'

  def Step(self, action, paths=[]):
    self._seq += 1

    msg = action + ' ' + str(self._seq)
    for path in paths:
      for x in path:
        msg += ' ' + str(int(x))
    logging.debug('Step %s', msg)
    self._sock.send(msg + '\n')

    line = self._fin.readline().strip()

    frame = Frame(line, self._prev_frames)

    assert frame.seq == self._seq, \
        'Expecting %d, got %d' % (self._seq, frame.seq)

    if frame.terminal:
      self._prev_frames.clear()
    else:
      self._prev_frames.append(frame)
      if len(self._prev_frames) > NUM_SNAPSHOTS-1:
        self._prev_frames.popleft()

    return frame

  def seq(self):
    return self._seq


def test_game():
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


def clipped_error(x):
  # Huber loss
  return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class NeuralNetwork:
  def __init__(self, name, trainable=True):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02), trainable=trainable)

    with tf.variable_scope(name):
      if not RAW_IMAGE:
        # Input 1.
        self.input1 = tf.placeholder(tf.float32, [None, INPUT_DIM-NIE])

        # Input 2.
        self.ies = tf.placeholder(tf.float32,
                                  [None, NUM_INCOMING_ENEMIES, 2*NUM_SNAPSHOTS])
        ies0 = tf.reshape(self.ies, [-1, 2*NUM_SNAPSHOTS])
        NIE1 = 8
        ies1 = tf.nn.relu(tf.matmul(ies0, var([2*NUM_SNAPSHOTS, NIE1])) +
                          var([NIE1]))
        ies2 = tf.nn.relu(tf.matmul(ies1, var([NIE1, NIE])) + var([NIE]))
        ies3 = tf.reshape(ies2, [-1, NUM_INCOMING_ENEMIES, NIE])
        input2 = tf.reduce_sum(ies3, axis = 1)

        self.input = tf.concat_v2([self.input1, input2], axis=1)
        if trainable:
          logging.info('input: %s', self.input.get_shape())

        N1 = 32
        N2 = 24
        N3 = 16

        fc1 = tf.nn.relu(tf.matmul(self.input, var([INPUT_DIM, N1])) +
                         var([N1]))

        fc2 = tf.nn.relu(tf.matmul(fc1, var([N1, N2])) + var([N2]))

        fc3 = tf.nn.relu(tf.matmul(fc2, var([N2, N3])) + var([N3]))

        self.value = tf.matmul(fc3, var([N3, 1])) + var([1])
        self.advantage = tf.matmul(fc3, var([N3, OUTPUT_DIM])) + var([OUTPUT_DIM])
        self.output = self.value + (self.advantage -
            tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
      else:
        # Input image.
        self.input = tf.placeholder(tf.float32,
            [None, SIDE, SIDE, NUM_SNAPSHOTS])
        if trainable:
          logging.info('input: %s', self.input.get_shape())

        # Conv 1.
        self.w1 = var([8, 8, NUM_SNAPSHOTS, 32])
        self.b1 = var([32])
        conv1 = tf.nn.relu(tf.nn.conv2d(
          self.input, self.w1, strides = [1, 4, 4, 1], padding = "VALID")
          + self.b1)
        if trainable:
          logging.info('conv1: %s', conv1.get_shape())

        # Conv 2.
        self.w2 = var([4, 4, 32, 64])
        self.b2 = var([64])
        conv2 = tf.nn.relu(tf.nn.conv2d(
          conv1, self.w2, strides = [1, 2, 2, 1], padding = "VALID")
          + self.b2)
        if trainable:
          logging.info('conv2: %s', conv2.get_shape())

        # Conv 3.
        self.w3 = var([3, 3, 64, 64])
        self.b3 = var([64])
        conv3 = tf.nn.relu(tf.nn.conv2d(
          conv2, self.w3, strides = [1, 1, 1, 1], padding = "VALID")
          + self.b3)
        if trainable:
          logging.info('conv3: %s', conv3.get_shape())

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
    assert len(self.theta) == (10 if RAW_IMAGE else 14), len(self.theta)

    if trainable:
      # Training.
      self.action = tf.placeholder(tf.int32, [None])
      action_one_hot = tf.one_hot(self.action, OUTPUT_DIM)
      self.y = tf.placeholder(tf.float32, [None])
      q_action = tf.reduce_sum(tf.multiply(self.output, action_one_hot),
          reduction_indices = 1)
      self.cost = tf.reduce_mean(clipped_error(q_action - self.y))
      self.optimizer = tf.train.RMSPropOptimizer(
          learning_rate=0.00025, momentum=.95, epsilon=1e-2).minimize(self.cost)

  def Vars(self):
    return self.theta

  def EvalAll(self, frames):
    return tf.get_default_session().run(
        [self.output, self.value, self.advantage], {
            self.input1: [f.datax for f in frames],
            self.ies: [f.ies for f in frames]
        })

  def Eval(self, frames):
    return self.EvalAll(frames)[0]

  def Train(self, tnn, mini_batch):
    frame_batch = [d[0] for d in mini_batch]
    frame1_batch = [d[1] for d in mini_batch]
    action_batch = [f.action_id for f in frame1_batch]

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
        self.input1: [f.datax for f in frame_batch],
        self.ies: [f.ies for f in frame_batch],
        self.action: action_batch,
        self.y: y_batch,
    }
    self.optimizer.run(feed_dict)
    return self.cost.eval(feed_dict)

  def CopyFrom(self, sess, src):
    for v1, v2 in zip(self.Vars(), src.Vars()):
      sess.run(v1.assign(v2))

  def CheckSum(self):
    return [np.sum(var.eval()) for var in self.Vars()]


class PathNeuralNetwork:
  def __init__(self, name):
    var = lambda shape: tf.Variable(tf.truncated_normal(shape, stddev=.02))

    with tf.variable_scope(name):
      INPUT_SIZE = 6
      OUTPUT_SIZE = 2*PATH_LEN
      LSTM_SIZE = 16
      self.input = tf.placeholder(tf.float32, [None, PATH_LEN, INPUT_SIZE])
      self.keep_prob = tf.placeholder(tf.float32)
      lstm0 = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, state_is_tuple=True)
      lstm1 = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, state_is_tuple=True)
      lstm1 = tf.nn.rnn_cell.DropoutWrapper(lstm1,
          output_keep_prob=self.keep_prob)
      lstm = tf.nn.rnn_cell.MultiRNNCell([lstm0, lstm1],
          state_is_tuple=True)
      logging.info('lstm.state_size: %s', lstm.state_size)
      logging.info('lstm.output_size: %s', lstm.output_size)
      rnn_out, state = tf.nn.dynamic_rnn(lstm, self.input, dtype=tf.float32)
      logging.info('rnn_out: %s', rnn_out.get_shape())
      rnn_out = tf.transpose(rnn_out, [1, 0, 2])
      rnn_out = rnn_out[-1]
      self.output = tf.matmul(rnn_out, var([LSTM_SIZE, OUTPUT_SIZE])) \
          + var([OUTPUT_SIZE])

      self.target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
      self.cost = tf.reduce_mean(tf.square(self.output - self.target))
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=5e-3, epsilon=1e-2).minimize(self.cost)

    self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    assert len(self.theta) == 6, len(self.theta)

  def Vars(self):
    return self.theta

  def Eval(self, inputs):
    if not inputs:
      return []
    outputs = self.output.eval({self.input: inputs, self.keep_prob: 1.0})
    outputs = np.round(outputs)
    return outputs

  @staticmethod
  def _EncodePathInput(frames, eid):
    pin = []
    pe = None
    frames = frames[-PATH_LEN:]
    for f in frames:
      e = f.incoming_enemies.get(eid)
      if not e or e.y < 72:
        pin = []
        pe = None
        continue
      vx = 0
      vy = 0
      if pe:
        vx = e.x - pe.x
        vy = e.y - pe.y
      if abs(vx) > 20 or abs(vy) > 20:
        break
      dx = e.x - f.galaxian.x
      coor = [dx, vx, vy]
      coor += one_hot(3, enemy_type(e.row))
      pin.append(coor)
      pe = e
    if not pin:
      return None
    assert len(pin) <= PATH_LEN, '{} {}'.format(len(pin), len(frames))
    pin = [pin[0]] * (PATH_LEN - len(pin)) + pin
    return np.array(pin)

  @staticmethod
  def EncodePathInputs(frames):
    cur = frames[-1]
    ret = {}
    for eid in cur.incoming_enemies:
      pin = PathNeuralNetwork._EncodePathInput(frames, eid)
      if pin is not None:
        ret[eid] = pin
    return ret

  @staticmethod
  def EncodePathData(frames):
    assert len(frames) >= 2*PATH_LEN

    data = []
    in_frames = frames[:-PATH_LEN]
    out_frames = frames[-PATH_LEN:]
    cur = in_frames[-1]
    latest = out_frames[-1]
    for eid in latest.incoming_enemies:
      if eid not in cur.incoming_enemies: continue

      pin = PathNeuralNetwork._EncodePathInput(in_frames, eid)
      if pin is None: continue

      pout = []
      pe = cur.incoming_enemies[eid]
      for f in out_frames:
        if eid not in f.incoming_enemies:
          break
        e = f.incoming_enemies[eid]
        vx = e.x - pe.x
        vy = e.y - pe.y
        if abs(vx) > 20 or abs(vy) > 20:
          break
        pe = e
        pout.append([vx, vy])
      if len(pout) < 5:
        continue
      if len(pout) < PATH_LEN:
        pout += [pout[-1]] * (PATH_LEN - len(pout))
      pout = sum(pout, [])
      pout = np.array(pout)

      data.append((pin, pout))

    return data

  @staticmethod
  def DecodePathOutputs(outputs, eids, cur):
    assert len(eids) == len(outputs), '{} {}'.format(len(eids), len(outputs))

    paths = []
    for eid, out in zip(eids, outputs):
      e = cur.incoming_enemies[eid]
      x = e.x
      y = e.y
      path = []
      for i in xrange(0, len(out), 2):
        x += out[i]
        y += out[i+1]
        path += [x, y]
      paths.append(path)
    return paths

  def Train(self, data):
    inputs = np.array([d[0] for d in data])
    targets = np.array([d[1] for d in data])
    feed_dict = {self.input: inputs, self.target: targets, self.keep_prob: 0.9}
    self.optimizer.run(feed_dict)
    return self.cost.eval(feed_dict)

  def Test(self, data):
    inputs = np.array([d[0] for d in data])
    targets = np.array([d[1] for d in data])

    feed_dict = {self.input: inputs, self.target: targets, self.keep_prob: 1.0}
    cost = self.cost.eval(feed_dict)
    outputs = self.output.eval(feed_dict)
    outputs = np.round(outputs)

    return cost, inputs, targets, outputs

  def CheckSum(self):
    return [np.sum(var.eval()) for var in self.Vars()]


def format_list(l):
  return '[' + ' '.join(['%6.2f' % x for x in l]) + ']'


class SavedVar:
  def __init__(self, init_val, name):
    self.var = tf.Variable(init_val, trainable=False, name=name)
    self.input = tf.placeholder(self.var.dtype, self.var.get_shape())
    self.assign = self.var.assign(self.input)

  def Eval(self):
    return self.var.eval()

  def Assign(self, sess, val):
    sess.run(self.assign, {self.input: val})


class SimpleSaver:
  def __init__(self, name, var_list, model_dir, filename):
    self.name = name
    self.model_dir = model_dir
    self.path = os.path.join(model_dir, filename)
    self.saver = tf.train.Saver(var_list)
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

  def Restore(self, sess):
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(sess, ckpt.model_checkpoint_path)
      logging.info("%s: Restored from %s", self.name,
          ckpt.model_checkpoint_path)
    else:
      logging.info("%s: No checkpoint found", self.name)

  def Save(self, sess, global_step=None):
    save_path = self.saver.save(sess, self.path, global_step=global_step)
    logging.info("%s: Saved to %s", self.name, save_path)


def main(unused_argv):
  logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO,
      format='%(message)s')

  port = FLAGS.port
  if FLAGS.server:
    server = subprocess.Popen([FLAGS.server, FLAGS.rom, str(port)])
    time.sleep(1)

  memory = deque()
  nn = NeuralNetwork('nn')
  tnn = NeuralNetwork('tnn', trainable=False)

  saved_step = SavedVar(0, 'step')
  saved_epsilon = SavedVar(INITIAL_EPSILON, 'epsilon')

  episode = deque()
  pdata = []
  pnn = PathNeuralNetwork('pnn')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = SimpleSaver('saver', nn.Vars() + [saved_step.var,
      saved_epsilon.var], FLAGS.checkpoint_dir, CHECKPOINT_FILE)
    saver.Restore(sess)

    tnn.CopyFrom(sess, nn)

    step = saved_step.Eval()
    epsilon = FLAGS.eps or saved_epsilon.Eval()

    pnn_saver = SimpleSaver('pnn', pnn.Vars(), FLAGS.pnn_dir, CHECKPOINT_FILE)
    pnn_saver.Restore(sess)

    cost = 0
    p_test_cost = 0
    p_train_cost = 0
    value = 0
    advantage = []

    game = Game(port)
    game.Start(step)
    frame = game.Step('_')

    action_summary = defaultdict(int)

    while True:
      if random.random() <= epsilon:
        rand = True
        action = ACTION_NAMES[random.randrange(OUTPUT_DIM)]
        value = 0
        advantage = []
      else:
        rand = False
        qs, values, advantages = nn.EvalAll([frame])
        q, value, advantage = qs[0], values[0], advantages[0]
        action = ACTION_NAMES[np.argmax(q)]
        action_summary[action] += 1

      paths = []
      if FLAGS.send_paths and episode:
        inputs_map = PathNeuralNetwork.EncodePathInputs(list(episode))
        pouts = pnn.Eval(inputs_map.values())
        paths = PathNeuralNetwork.DecodePathOutputs(
            pouts, inputs_map.keys(), episode[-1])

      frame1 = game.Step(action, paths=paths)
      step = game.seq()

      if not frame.terminal:
        memory.append((frame, frame1))
        if len(memory) > REPLAY_MEMORY:
          memory.popleft()

      episode.append(frame1)
      if len(episode) > 2*PATH_LEN:
        episode.popleft()
      if len(episode) >= 2*PATH_LEN and FLAGS.train_paths:
        pdata.extend(PathNeuralNetwork.EncodePathData(list(episode)))
      if frame1.terminal:
        episode.clear()

      if step % PATH_TEST_INTERVAL == 0 and pdata:
        p_test_cost, inputs, targets, outputs = pnn.Test(pdata[-5:])
        logging.info(
            'pnn test: cost: %s\n '
            'inputs:\n%s\n targets:\n%s\n outputs:\n%s\n delta:\n%s',
            p_test_cost, inputs, targets, outputs, outputs - targets)

      if step % TRAIN_INTERVAL == 0 and step > OBSERVE_STEPS:
        mini_batch = random.sample(memory, min(len(memory), MINI_BATCH_SIZE))
        if memory:
          mini_batch.append(memory[-1])
        cost = nn.Train(tnn, mini_batch)

        if len(pdata) >= 1000:
          EPOCHS = 2
          p_train_cost = 0
          n = 0
          for e in xrange(EPOCHS):
            logging.info('pnn train: epoch %d', e)
            for i in xrange(0, len(pdata), MINI_BATCH_SIZE):
              p_train_cost += pnn.Train(pdata[i: i+MINI_BATCH_SIZE])
              n += 1
          p_train_cost /= n
          logging.info('pnn train: data size: %d cost: %s', len(pdata),
              p_train_cost)
          pdata = []

      frame = frame1

      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if step % UPDATE_TARGET_NETWORK_INTERVAL == 0:
        logging.info(action_summary)
        action_summary.clear()

        logging.info('Target network before: %s', tnn.CheckSum())
        tnn.CopyFrom(sess, nn)
        logging.info('Target network after: %s', tnn.CheckSum())

      if step % SAVE_INTERVAL == 0:
        saved_step.Assign(sess, step)
        saved_epsilon.Assign(sess, epsilon)
        saver.Save(sess, global_step=step)
        pnn_saver.Save(sess, global_step=step)

      logging.info(
          "Step %d eps: %.6f value: %7.3f adv: %-43s action: %s%s "
          "reward: %5.2f %s pnn: %s p_train_cost: %.6f p_test_cost: %.6f" %
          (step, epsilon, value, format_list(advantage), frame1.action,
            ' ?'[rand], frame1.reward, ' t'[frame1.terminal],
            format_list(pnn.CheckSum()), p_train_cost, p_test_cost))


if __name__ == '__main__':
  tf.app.run()
