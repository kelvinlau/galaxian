"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.tithx7juq

TODO: Save png to verify input data.
TODO: Training the model only on score increases?
TODO: Dropout/Bayesian.
TODO: A3C.
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
import scipy.signal
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', False, 'enable logging.debug')
flags.DEFINE_string('server', '', 'server binary')
flags.DEFINE_string('rom', './galaxian.nes', 'galaxian nes rom file')
flags.DEFINE_string('checkpoint_dir', 'models/2.28', 'checkpoint dir')
flags.DEFINE_integer('port', 62343, 'server port to conenct')
flags.DEFINE_bool('train_paths', False, 'train pnn')
flags.DEFINE_bool('send_paths', False, 'send path to render by lua server')
flags.DEFINE_string('pnn_dir', 'models/pnn10', 'pnn model dir')


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
  INPUT_DIM = 3 + (2*FOCUS+3) + NUM_INCOMING_ENEMIES*5 + 2*WIDTH
PATH_LEN = 12
ACTION_NAMES = ['_', 'L', 'R', 'A', 'l', 'r']
ACTION_ID = {ACTION_NAMES[i]: i for i in xrange(len(ACTION_NAMES))}
OUTPUT_DIM = len(ACTION_NAMES)


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
      self.data.append(self.missile.y / 200.0 if self.missile.y < 200 else 0)
      logging.debug('missile %d,%d', self.missile.x, self.missile.y)

      # galaxian x
      self.data.append(galaxian.x / 256.0)

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

      # incoming enemy x, y relative to galaxian, and enemy type.
      ies = []
      for eid, e in sorted(
          self.incoming_enemies.items(), key = lambda p: p[1].y):
        dx = (e.x - galaxian.x) / 256.0
        dy = (e.y - galaxian.y) / 256.0
        ies.append([dx, dy] + one_hot(3, enemy_type(e.row)))
      for i in xrange(NUM_INCOMING_ENEMIES-len(ies)):
        ies.append([1, 1, 0, 0, 0])
      ies = sum(ies, [])
      self.data.extend(ies)

      # hit map
      def ix(x):
        return max(0, min(WIDTH-1, (x-galaxian.x+128)/DX))
      # out-of-bound tiles have penality.
      imap = [0. if ix(0) <= i <= ix(255) else 1. for i in range(WIDTH)]
      bmap = imap[:]
      if prev_frames:
        steps = len(prev_frames)
        y1 = galaxian.y-8
        y2 = galaxian.y+8
        for eid, e in self.incoming_enemies.iteritems():
          pe = None  # the furthest frame having this enemy
          for pf in prev_frames:
            if eid in pf.incoming_enemies:
              pe = pf.incoming_enemies[eid]
              break
          x1, x2, t = None, None, None
          if pe and pe.y < e.y < y1:
            x1 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y1-pe.y)+pe.x))
            x2 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y2-pe.y)+pe.x))
            t = (y1-e.y)*1.0/(e.y-pe.y)*steps
          elif y1 <= e.y <= y2:
            x1 = x2 = e.x
            t = 0
          if x1 is not None:
            hit = max(0., 1.-t/24.)
            for i in xrange(ix(x1), ix(x2)+1):
              imap[i] += hit
        for eid, e in self.bullets.iteritems():
          pe = None  # the furthest frame having this bullet
          for pf in prev_frames:
            if eid in pf.bullets:
              pe = pf.bullets[eid]
              break
          x1, x2, t = None, None, None
          if pe and pe.y < e.y < y1:
            x1 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y1-pe.y)+pe.x))
            x2 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y2-pe.y)+pe.x))
            t = (y1-e.y)*1.0/(e.y-pe.y)*steps
          elif y1 <= e.y <= y2:
            x1 = x2 = e.x
            t = 0
          if x1 is not None:
            hit = max(0., 1.-t/12.)
            for i in xrange(ix(x1), ix(x2)+1):
              bmap[i] += hit
      self.data += imap
      self.data += bmap
      logging.debug('imap [%s]', ''.join(['x' if h>0 else '_' for h in imap]))
      logging.debug('bmap [%s]', ''.join(['x' if h>0 else '_' for h in bmap]))

      if not self.terminal:
        self.reward -= min(1., bmap[ix(galaxian.x)]) * .25

      assert len(self.data) == INPUT_DIM, \
          '{} vs {}'.format(len(self.data), INPUT_DIM)
      self.data = np.array(self.data)
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
        self.data = np.reshape(self.data, (SIDE, SIDE, 1))
        for i in xrange(NUM_SNAPSHOTS-1):
          self.data = np.append(
              self.data,
              np.reshape(self.data, (SIDE, SIDE, 1)),
              axis = 2)
      else:
        prev_frame = prev_frames[-1]
        self.data = np.append(
            np.reshape(self.data, (SIDE, SIDE, 1)),
            prev_frame.data[:, :, :NUM_SNAPSHOTS-1],
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
    return np.sum(self.data)


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


def categorical_sample(logits, n):
  logits = logits - tf.reduce_max(logits, [1], keep_dims=True)
  i = tf.squeeze(tf.multinomial(logits, 1), [1])
  return tf.one_hot(i, n)


# discount([1, 1, 1], .99) == [2.9701, 1.99, 1.]
def discount(x, gamma):
  return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


class ACNeuralNetwork:
  def __init__(self, name, trainable=True):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02), trainable=trainable)

    with tf.variable_scope(name):
      if not RAW_IMAGE:
        # Input.
        self.input = tf.placeholder(tf.float32, [None, INPUT_DIM], name='input')
        if trainable:
          logging.info('input: %s', self.input.get_shape())
        x = self.input

        N1 = 64
        N2 = 64
        x = tf.nn.elu(tf.matmul(x, var([INPUT_DIM, N1])) + var([N1]))
        x = tf.nn.elu(tf.matmul(x, var([N1, N2])) + var([N2]))
        x = tf.expand_dims(x, [0])

        LSTM_SIZE = 64
        lstm = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c')
        h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h')
        self.state_in = [c_in, h_in]

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=tf.nn.rnn_cell.LSTMStateTuple(c_in, h_in),
            sequence_length=tf.shape(self.input)[:1])
        lstm_c, lstm_h = lstm_state
        logging.info('lstm_outputs: %s', lstm_outputs.get_shape())
        logging.info('lstm_c: %s', lstm_c.get_shape())
        logging.info('lstm_h: %s', lstm_h.get_shape())
        self.state_out = [lstm_c, lstm_h]

        x = tf.reshape(lstm_outputs, [-1, LSTM_SIZE])
        self.logits = tf.matmul(x, var([LSTM_SIZE, OUTPUT_DIM])) \
            + var([OUTPUT_DIM])
        self.value = tf.reshape(
            tf.matmul(x, var([LSTM_SIZE, 1])) + var([1]),
            [-1])
        self.action = categorical_sample(self.logits, OUTPUT_DIM)
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
    assert len(self.theta) == (10 if RAW_IMAGE else 10), len(self.theta)

    if trainable:
      self.actual_action = tf.placeholder(tf.float32, [None, OUTPUT_DIM],
          name='actual_action')
      self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
      self.r = tf.placeholder(tf.float32, [None], name='r')

      log_prob = tf.nn.log_softmax(self.logits)
      prob = tf.nn.softmax(self.logits)

      policy_loss = -tf.reduce_sum(
          tf.reduce_sum(log_prob * self.actual_action, [1]) * self.advantage)
      value_loss = 0.5 * tf.reduce_sum(tf.square(self.value - self.r))
      entropy = -tf.reduce_sum(prob * log_prob)
      self.loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

      grads = tf.gradients(self.loss, self.theta)
      grads, _ = tf.clip_by_global_norm(grads, 40.0)

      self.optimizer = tf.train.AdamOptimizer(1e-4).apply_gradients(
          zip(grads, self.theta))

  def Vars(self):
    return self.theta

  def InitialState(self):
    return self.state_init

  def Eval(self, frame, state):
    ret = tf.get_default_session().run(
        [self.action, self.value] + self.state_out + [self.logits], {
            self.input: [frame.data],
            self.state_in[0]: state[0],
            self.state_in[1]: state[1],
        })
    return ret[0][0], ret[1][0], ret[2:4], ret[4][0]

  def Train(self, experience):
    GAMMA = 0.99
    LAMBDA = 1.0

    _, last_frame, _, last_state = experience[-1]
    terminal = last_frame.terminal
    last_value = self.Eval(last_frame, last_state)[1] if not terminal else 0.
    inputs = np.array([f.data for f, _, _, _ in experience])
    frames = [e[1] for e in experience]
    actions = np.array([one_hot(OUTPUT_DIM, f.action_id) for f in frames])
    rewards = np.array([f.reward for f in frames])
    values = np.array([e[2] for e in experience] + [last_value])
    rs = discount(np.append(rewards, [last_value]), GAMMA)[:-1]
    delta_t = rewards + GAMMA * values[1:] - values[:-1]
    advantages = discount(delta_t, GAMMA * LAMBDA)
    state = experience[0][3]

    self.optimizer.run({
      self.input: inputs,
      self.actual_action: actions,
      self.advantage: advantages,
      self.r: rs,
      self.state_in[0]: state[0],
      self.state_in[1]: state[1],
    })

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


def format_list(l, fmt='%6.2f'):
  return '[' + ' '.join([fmt % x for x in l]) + ']'


class SavedVar:
  def __init__(self, name, init_val):
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
    self.saver = tf.train.Saver(var_list, max_to_keep=1000,
        keep_checkpoint_every_n_hours=1, pad_step_number=True)
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
  TRAIN_LENGTH = 20
  PATH_TEST_INTERVAL = 1000
  SUMMARY_INTERVAL = 10000
  SAVE_INTERVAL = 10000
  CHECKPOINT_FILE = 'model.ckpt'

  logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO,
      format='%(message)s')

  port = FLAGS.port
  if FLAGS.server:
    server = subprocess.Popen([FLAGS.server, FLAGS.rom, str(port)])
    time.sleep(1)

  saved_step = SavedVar('step', 0)
  saved_ep = SavedVar('ep', 0)

  ac = ACNeuralNetwork('ac')
  state = ac.InitialState()
  experience = []
  length = 0
  rewards = 0

  pframes = deque()
  pdata = []
  pnn = PathNeuralNetwork('pnn')
  p_test_cost = 0
  p_train_cost = 0

  action_summary = defaultdict(int)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = SimpleSaver('saver', ac.Vars() + [saved_step.var],
        FLAGS.checkpoint_dir, CHECKPOINT_FILE)
    saver.Restore(sess)

    pnn_saver = SimpleSaver('pnn', pnn.Vars(), FLAGS.pnn_dir, CHECKPOINT_FILE)
    pnn_saver.Restore(sess)

    step = saved_step.Eval()
    ep = saved_ep.Eval()

    game = Game(port)
    game.Start(step)
    frame = game.Step('_')

    while True:
      # eval action
      action, value, state, logits = ac.Eval(frame, state)
      action = ACTION_NAMES[action.argmax()]

      # eval paths
      paths = []
      if FLAGS.send_paths and pframes:
        inputs_map = PathNeuralNetwork.EncodePathInputs(list(pframes))
        pouts = pnn.Eval(inputs_map.values())
        paths = PathNeuralNetwork.DecodePathOutputs(
            pouts, inputs_map.keys(), pframes[-1])

      # take action
      frame1 = game.Step(action, paths=paths)
      step = game.seq()

      experience.append((frame, frame1, value, state))
      length += 1
      rewards += frame1.reward

      frame = frame1

      logging.info(
          "Step %d value: %7.3f action: %s reward: %5.2f ac: %s logits: %s" %
          (step, value, frame.action, frame.reward, format_list(ac.CheckSum()),
            format_list(logits, fmt='%7.3f')))

      # pnn training
      pframes.append(frame)
      if len(pframes) > 2*PATH_LEN:
        pframes.popleft()

      if FLAGS.train_paths:
        if len(pframes) >= 2*PATH_LEN:
          pdata.extend(PathNeuralNetwork.EncodePathData(list(pframes)))

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

      # pnn testing
      if step % PATH_TEST_INTERVAL == 0 and pdata:
        p_test_cost, inputs, targets, outputs = pnn.Test(pdata[-5:])
        logging.info(
            'pnn test: cost: %s\n '
            'inputs:\n%s\n targets:\n%s\n outputs:\n%s\n delta:\n%s',
            p_test_cost, inputs, targets, outputs, outputs - targets)

      # policy training
      if len(experience) >= TRAIN_LENGTH or frame.terminal:
        ac.Train(experience)
        experience = []

      # reset on terminal
      if frame.terminal:
        logging.info('episode %d: length: %d rewards: %f', ep, length, rewards)
        if length > 3600:
          logging.warn('suspicious long episode of length %d, bug?', length)
        ep += 1
        frame = game.Step('_')
        state = ac.InitialState()
        length = 0
        rewards = 0
        pframes.clear()

      # summary
      action_summary[frame.action] += 1
      if step % SUMMARY_INTERVAL == 0:
        logging.info('actions %s', action_summary)
        action_summary.clear()

      # save models
      if step % SAVE_INTERVAL == 0:
        saved_step.Assign(sess, step)
        saved_ep.Assign(sess, ep)
        saver.Save(sess, global_step=step)
        pnn_saver.Save(sess, global_step=step)


if __name__ == '__main__':
  tf.app.run()
