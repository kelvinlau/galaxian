"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.tithx7juq
https://github.com/openai/universe-starter-agent/

TODO: Add var names, print shapes.
TODO: ac-0.
TODO: No artificial rewards.
TODO: Raw image as input.
TODO: Save png to verify input data.
TODO: Dropout/Bayesian.
TODO: Model based, Dyna, Sarsa, TD search, Monte Carlo.
TODO: Fix A holding.
TODO: Tree trimming.
"""

from __future__ import print_function
from collections import deque
import os
import random
import time
import math
import socket
import subprocess
import logging
import threading
import copy
#import cv2
import scipy.signal
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', False, 'enable logging.debug')
flags.DEFINE_bool('log_steps', False, 'log steps')
flags.DEFINE_string('server_tasks', '', 'tasks to start servers')
flags.DEFINE_string('server', './server', 'server binary')
flags.DEFINE_string('rom', './galaxian.nes', 'galaxian nes rom file')
flags.DEFINE_string('logdir', 'logs/2.28', 'Supervisor logdir')
flags.DEFINE_integer('port', 5001, 'server port to conenct')
flags.DEFINE_integer('num_workers', 1, 'num servers')
flags.DEFINE_bool('train', False, 'train or just play')
flags.DEFINE_bool('train_pnn', False, 'train pnn')
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
  HMAP_WIDTH = 256 / DX
  FOCUS = 16
  INPUT_DIM = 3 + (2*FOCUS+3) + NUM_INCOMING_ENEMIES*5 + 2*HMAP_WIDTH
PATH_LEN = 12
ACTIONS = ['_', 'L', 'R', 'A', 'l', 'r']
ACTION_ID = {ACTIONS[i]: i for i in xrange(len(ACTIONS))}
OUTPUT_DIM = len(ACTIONS)


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


def rehash(x, y):
  return (x*13 + y) % (2**64)


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __hash__(self):
    return rehash(hash(self.x), hash(self.y))

  def __repr__(self):
    return '{},{}'.format(self.x, self.y)

  def __str__(self):
    return '{},{}'.format(self.x, self.y)


def dist(a, b):
  return max(abs(a.x - b.x), abs(a.y - b.y))


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


def hmap_string(hmap):
  def s(h):
    if h <= 0: return '_'
    if h >= 1: return 'x'
    return str(round(h * 10))[0]
  return ''.join([s(h) for h in hmap])


def enemy_type(row):
  assert 0 <= row <= 5
  if row <= 2:
    return 0
  elif row == 3:
    return 1
  else:
    return 2


class Frame:
  def __init__(self, line, prev_frames, pnn):
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

    self.still_enemies = []
    for i, mask in enumerate(self.masks):
      x = self.sdx + 16 * i
      y = 108
      while mask:
        if mask % 2:
          self.still_enemies.append(Point(x, y))
        mask /= 2
        y -= 12
    assert len(self.still_enemies) <= 48, len(self.still_enemies)

    self.incoming_enemies = {}
    for i in xrange(self.NextInt()):
      eid = self.NextInt()
      e = self.NextPoint()
      e.row = self.NextInt()
      self.incoming_enemies[eid] = e
    self.dead = {}
    if prev_frames:
      pf = prev_frames[-1]
      for eid, e in self.incoming_enemies.items():
        pe = pf.incoming_enemies.get(eid)
        if pe is not None and pe.x == e.x and pe.y == e.y or eid in pf.dead:
          del self.incoming_enemies[eid]
          self.dead[eid] = 1

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
      self.svx = svx
      self.data += smap
      self.data.append(sl)
      self.data.append(sr)
      self.data.append(svx)
      logging.debug('smap [%s] %s %s %s', hmap_string(smap), sl, sr, svx)

      # incoming enemies: dx, dy, type.
      # TODO: index by eid?
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
        return max(0, min(HMAP_WIDTH-1, (int(round(x))-galaxian.x+128)/DX))
      # out-of-bound tiles have penality.
      imap = [0. if ix(0) <= i <= ix(255) else 1. for i in range(HMAP_WIDTH)]
      bmap = imap[:]

      # incoming enemy paths
      self.paths = {}
      y1 = galaxian.y-8
      y2 = galaxian.y+8
      pins_map = PathNeuralNetwork.EncodePathInputs(list(prev_frames)+[self])
      pouts = pnn.Eval(pins_map.values())
      self.paths = PathNeuralNetwork.DecodePathOutputs(
          pouts, pins_map.keys(), self.incoming_enemies)
      for eid, e in self.incoming_enemies.iteritems():
        path = self.paths.get(eid)
        if path is None: continue
        for t, p in enumerate(path):
          if y1 <= p.y <= y2:
            imap[ix(p.x)] += max(0., 1-t/24.)

      # bullets
      self.bv = {}
      for eid, e in self.bullets.iteritems():
        pe = None  # the furthest frame having this bullet
        steps = 0
        for pf in reversed(prev_frames):
          if eid in pf.bullets:
            pe = pf.bullets[eid]
            steps += 1
          else:
            break
        x1, x2, t = None, None, None
        if pe and pe.y < e.y < y1:
          x1 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y1-pe.y)+pe.x))
          x2 = int(round((e.x-pe.x)*1.0/(e.y-pe.y)*(y2-pe.y)+pe.x))
          if x1 > x2:
            x1, x2 = x2, x1
          t = (y1-e.y)*1.0/(e.y-pe.y)*steps
        elif y1 <= e.y <= y2:
          x1 = x2 = e.x
          t = 0
        if x1 is not None:
          hit = max(0., 1.-t/12.)
          for i in xrange(ix(x1), ix(x2)+1):
            bmap[i] += hit
        if pe:
          self.bv[eid] = Point((e.x-pe.x)/steps, (e.y-pe.y)/steps)
        else:
          self.bv[eid] = Point(0, 1)

      self.data += imap
      self.data += bmap
      logging.debug('imap [%s]', hmap_string(imap))
      logging.debug('bmap [%s]', hmap_string(bmap))

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

      for e in self.still_enemies:
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


def is_dangerous(frame):
  for e in frame.incoming_enemies.values():
    if e.y > 150:
      return True
  for b in frame.bullets.values():
    if b.y > 150:
      return True
  return False


class SFrame:
  def __init__(self, src):
    assert isinstance(src, Frame)
    self.galaxian = src.galaxian
    self.missile = src.missile
    self.still_enemies = src.still_enemies
    self.incoming_enemies = src.incoming_enemies
    self.bullets = src.bullets
    self.seq = src.seq
    self.rewards = 0

  def __hash__(self):
    h = hash(self.seq)
    h = rehash(h, hash(self.galaxian.x))
    h = rehash(h, hash(self.missile))
    for eid in self.incoming_enemies:
      h = rehash(h, hash(eid))
    return h


class Game:
  def __init__(self, port, pnn):
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect(('localhost', port))
    self._fin = self._sock.makefile()
    self._pnn = pnn
    self.last_frames = deque()
    self.length = 0
    self.rewards = 0

  def Start(self, seq=0):
    self._seq = seq
    self._sock.send('galaxian:start %d\n' % (seq+1))
    assert self._fin.readline().strip() == 'ack'

  def Step(self, action, paths=[]):
    if self.last_frames and self.last_frames[-1].terminal:
      self.last_frames.clear()
      self.length = 0
      self.rewards = 0

    self._seq += 1

    msg = action + ' ' + str(self._seq)
    for path in paths:
      for p in path:
        msg += ' ' + str(int(p.x))
        msg += ' ' + str(int(p.y))
    logging.debug('Step %s', msg)
    self._sock.send(msg + '\n')

    line = self._fin.readline().strip()

    frame = Frame(line, self.last_frames, self._pnn)

    assert frame.seq == self._seq, \
        'Expecting %d, got %d' % (self._seq, frame.seq)

    self.last_frames.append(frame)
    if len(self.last_frames) > 2*PATH_LEN:
      self.last_frames.popleft()
    self.length += 1
    self.rewards += frame.reward

    return frame

  def Simulate(self, s, action, decay):
    FRAME_SKIP = 5
    WIDTH = 256

    frame = self.last_frames[-1]
    dep = s.seq - frame.seq

    t = copy.deepcopy(s)
    t.seq += 1

    # still enemies' speed is 1/5.
    for e in t.still_enemies:
      e.x += frame.svx * 1

    fired = 0
    for i in xrange(1, FRAME_SKIP+1):
      if action in ['L', 'l']:
        t.galaxian.x = max(t.galaxian.x - 1, 22)
      elif action in ['R', 'r']:
        t.galaxian.x = min(t.galaxian.x + 1, WIDTH-22)

      if t.missile.y < 200 or fired:
        t.missile.y -= 4
        if t.missile.y < 0:
          t.missile.y = 200
        fired = 0
      elif action in ['A', 'l', 'r'] and i == 1:
        fired = 1

      if t.missile.y >= 200:
        t.missile.x = t.galaxian.x

      for eid, e in t.bullets.iteritems():
        se = s.bullets[eid]
        v = frame.bv[eid]
        e.x = int(round(se.x + v.x * i / FRAME_SKIP))
        e.y = int(round(se.y + v.y * i / FRAME_SKIP))

      for eid, e in t.incoming_enemies.iteritems():
        se = s.incoming_enemies[eid]
        path = frame.paths.get(eid)
        if path:
          te = path[dep]
        else:
          te = se
        e.x = int(round(se.x + (te.x - se.x) * i / FRAME_SKIP))
        e.y = int(round(se.y + (te.y - se.y) * i / FRAME_SKIP))

      for b in t.bullets.values():
        if dist(b, t.galaxian) <= 8:
          t.rewards = -100 * decay
          return t

      for e in t.incoming_enemies.values():
        if dist(e, t.galaxian) <= 12:
          t.rewards = -100 * decay
          return t
      
      if t.missile.y < 200:
        hits = 0
        if t.missile < 112:
          num_still_enemies = len(t.still_enemies)
          t.still_enemies = filter(lambda e: dist(e, t.missile) > 4,
              t.still_enemies)
          hits += num_still_enemies - len(t.still_enemies)
          if hits:
            t.rewards += 1 * hits * decay
        
        for eid, e in t.incoming_enemies.items():
          if dist(e, t.missile) <= 4:
            t.rewards += (enemy_type(e.row) + 2) * decay
            del t.incoming_enemies[eid]
            hits += 1

        if hits:
          t.missile.y = 200
          t.missile.x = t.galaxian.x

    return t

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
  def __init__(self, name, global_ac=None):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02))

    with tf.variable_scope(name):
      if not RAW_IMAGE:
        # Input.
        self.input = tf.placeholder(tf.float32, [None, INPUT_DIM], name='input')
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

        # Conv 1.
        self.w1 = var([8, 8, NUM_SNAPSHOTS, 32])
        self.b1 = var([32])
        conv1 = tf.nn.relu(tf.nn.conv2d(
          self.input, self.w1, strides = [1, 4, 4, 1], padding = "VALID")
          + self.b1)

        # Conv 2.
        self.w2 = var([4, 4, 32, 64])
        self.b2 = var([64])
        conv2 = tf.nn.relu(tf.nn.conv2d(
          conv1, self.w2, strides = [1, 2, 2, 1], padding = "VALID")
          + self.b2)

        # Conv 3.
        self.w3 = var([3, 3, 64, 64])
        self.b3 = var([64])
        conv3 = tf.nn.relu(tf.nn.conv2d(
          conv2, self.w3, strides = [1, 1, 1, 1], padding = "VALID")
          + self.b3)

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

      self.var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
          scope=name)
      assert len(self.var_list) == (10 if RAW_IMAGE else 10), len(self.var_list)

      if global_ac is not None:
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

        grads = tf.gradients(self.loss, self.var_list)
        grads_clipped, _ = tf.clip_by_global_norm(grads, 40.0)

        self.optimizer = tf.train.AdamOptimizer(1e-4).apply_gradients(
            zip(grads_clipped, global_ac.var_list))

        self.sync = tf.group(*[dst.assign(src)
          for dst, src in zip(self.var_list, global_ac.var_list)])

        batch_size = tf.to_float(tf.shape(self.input)[0])
        summaries = [
          tf.summary.scalar("policy_loss", policy_loss / batch_size),
          tf.summary.scalar("value_loss", value_loss / batch_size),
          tf.summary.scalar("entropy", entropy / batch_size),
          tf.summary.scalar("loss", self.loss / batch_size),
          tf.summary.scalar("grad_global_norm", tf.global_norm(grads)),
          tf.summary.scalar("var_global_norm",
              tf.global_norm(self.var_list)),
        ]
        self.summary_op = tf.summary.merge(summaries)

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

  def Train(self, experience, return_summary=False):
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

    ops = []
    if return_summary:
      ops.append(self.summary_op)
    ops.append(self.optimizer)

    results = tf.get_default_session().run(ops, {
      self.input: inputs,
      self.actual_action: actions,
      self.advantage: advantages,
      self.r: rs,
      self.state_in[0]: state[0],
      self.state_in[1]: state[1],
    })

    if return_summary:
      return results[0]

  def Sync(self):
    tf.get_default_session().run(self.sync)

  def CheckSum(self):
    return [np.sum(var.eval()) for var in self.var_list]


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
      rnn_out, state = tf.nn.dynamic_rnn(lstm, self.input, dtype=tf.float32)
      rnn_out = tf.transpose(rnn_out, [1, 0, 2])
      rnn_out = rnn_out[-1]
      self.output = tf.matmul(rnn_out, var([LSTM_SIZE, OUTPUT_SIZE])) \
          + var([OUTPUT_SIZE])

      self.target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
      self.cost = tf.reduce_mean(tf.square(self.output - self.target))
      self.optimizer = tf.train.AdamOptimizer(
          learning_rate=5e-3, epsilon=1e-2).minimize(self.cost)

    self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=name)
    assert len(self.var_list) == 6, len(self.var_list)

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
    for f in frames:
      e = f.incoming_enemies.get(eid)
      if not e or e.y < 72:  # TODO: include <72
        pin = []
        pe = None
        continue
      vx = 0
      vy = 0
      if pe:
        vx = e.x - pe.x
        vy = e.y - pe.y
      if abs(vx) > 20 or abs(vy) > 20:
        pin = []
        pe = None
        continue
      dx = e.x - f.galaxian.x
      coor = [dx, vx, vy]  # TODO: Add dy.
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
    frames = frames[-PATH_LEN:]
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
    in_frames = frames[-2*PATH_LEN:-PATH_LEN]
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
  def DecodePathOutputs(outputs, eids, incoming_enemies):
    assert len(eids) == len(outputs), '{} {}'.format(len(eids), len(outputs))

    paths = {}
    for eid, out in zip(eids, outputs):
      e = incoming_enemies[eid]
      x = e.x
      y = e.y
      path = []
      for i in xrange(0, len(out), 2):
        x += out[i]
        y += out[i+1]
        path.append(Point(x, y))
      paths[eid] = path
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
    return [np.sum(var.eval()) for var in self.var_list]


def format_list(l, fmt='%6.2f'):
  return '[' + ' '.join([fmt % x for x in l]) + ']'


def parse_ints(s):
  if not s:
    return []
  ret = []
  for t in s.split(','):
    if '-' in t:
      a, b = map(int, t.split('-'))
    else:
      a = b = int(t)
    ret += range(a, b+1)
  return ret


class SavedVar:
  def __init__(self, name, init_val):
    self.var = tf.Variable(init_val, trainable=False, name=name)
    self.val = tf.placeholder(self.var.dtype, self.var.get_shape())
    self.assign = self.var.assign(self.val)
    self.inc = self.var.assign_add(self.val)

  def Eval(self):
    return self.var.eval()

  def Assign(self, val):
    tf.get_default_session().run(self.assign, {self.val: val})

  def Inc(self, delta):
    tf.get_default_session().run(self.inc, {self.val: delta})


def main(unused_argv):
  logging.basicConfig(level=logging.DEBUG if FLAGS.debug else logging.INFO,
      format='%(message)s')

  global_step = SavedVar('step', 0)
  global_ac = ACNeuralNetwork('ac')
  pnn = PathNeuralNetwork('pnn')
  summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.logdir, 'summary'))

  server_tasks = parse_ints(FLAGS.server_tasks)
  workers = [
      Worker(global_step, global_ac, pnn, summary_writer, i, i in server_tasks)
      for i in xrange(16)]

  sv = tf.train.Supervisor(logdir=FLAGS.logdir,
                           global_step=global_step.var,
                           saver=tf.train.Saver(
                               max_to_keep=1000,
                               keep_checkpoint_every_n_hours=1,
                               pad_step_number=True),
                           summary_op=None,
                           summary_writer=summary_writer,
                           save_model_secs=600,
                           save_summaries_secs=60)

  config = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.num_workers,
      log_device_placement=False)

  with sv.managed_session(config=config) as sess, sess.as_default():
    logging.info('ac: %s', format_list(global_ac.CheckSum()))
    logging.info('pnn: %s', format_list(pnn.CheckSum()))

    for worker in workers[:FLAGS.num_workers]:
      worker.Start(sv, sess)

    while any(w.is_alive() for w in workers):
      time.sleep(1)


class Worker(threading.Thread):
  def __init__(self, global_step, global_ac, pnn, summary_writer, task_id,
               start_server):
    threading.Thread.__init__(self, name='worker-'+str(task_id))
    self.daemon = True

    self.global_step = global_step
    self.pnn = pnn
    self.summary_writer = summary_writer
    self.task_id = task_id
    self.start_server = start_server
    self.ac = ACNeuralNetwork('ac_' + str(task_id), global_ac=global_ac)

  def Start(self, sv, sess):
    self.sv = sv
    self.sess = sess
    self.start()

  def run(self):
    port = FLAGS.port + self.task_id
    logging.info('starting worker %d at port %d', self.task_id, port)

    if self.start_server:
      server = subprocess.Popen([FLAGS.server, FLAGS.rom, str(port)],
              stdout=open('/tmp/galaxian-{}.stdout'.format(self.task_id), 'w'),
              stderr=open('/tmp/galaxian-{}.stderr'.format(self.task_id), 'w'))
      time.sleep(5)

    game = self.game = Game(port, self.pnn)
    game.Start()
    frame = game.Step('_')

    ac = self.ac
    state = ac.InitialState()
    experience = []

    train_pnn = FLAGS.train_pnn and self.task_id == 0
    if train_pnn:
      pdata = []
      p_test_cost = 0
      p_train_cost = 0

    lvl = logging.INFO if FLAGS.log_steps else logging.DEBUG
    step = 0
    trainings = 0
    action_summary = [0] * OUTPUT_DIM

    sess = self.sess
    with sess.as_default():
      ac.Sync()

      while not self.sv.should_stop():
        # eval action
        action, value, state, logits = ac.Eval(frame, state)
        action = ACTIONS[action.argmax()]
        if is_dangerous(frame):
          action = self.plan(logits)
        paths = frame.paths.values() if FLAGS.send_paths else []

        # take action
        frame1 = game.Step(action, paths=paths)
        experience.append((frame, frame1, value, state))
        frame = frame1

        step += 1
        logging.log(lvl,
            "Step %d value: %7.3f logits: %s action: %s reward: %5.2f",
            step, value, format_list(logits, fmt='%7.3f'), frame.action,
            frame.reward)

        # policy training
        TRAIN_LENGTH = 20
        if FLAGS.train and len(experience) >= TRAIN_LENGTH or frame.terminal:
          trainings += 1
          do_summary = self.task_id == 0 and trainings % 10 == 0

          ac.Sync()
          summary = ac.Train(experience, return_summary=do_summary)
          self.global_step.Inc(len(experience))
          experience = []

          if do_summary:
            self.summary_writer.add_summary(
                tf.Summary.FromString(summary),
                self.global_step.Eval())

        # pnn training
        if train_pnn:
          if len(game.last_frames) >= 2*PATH_LEN:
            pdata.extend(PathNeuralNetwork.EncodePathData(
              list(game.last_frames)))

          # training
          if len(pdata) >= 1000:
            EPOCHS = 2
            p_train_cost = 0
            n = 0
            for e in xrange(EPOCHS):
              logging.info('pnn train: epoch %d', e)
              for i in xrange(0, len(pdata), MINI_BATCH_SIZE):
                p_train_cost += self.pnn.Train(pdata[i: i+MINI_BATCH_SIZE])
                n += 1
            p_train_cost /= n
            logging.info('pnn train: data size: %d cost: %s', len(pdata),
                p_train_cost)
            pdata = []

          # testing
          if step % 10000 == 0 and pdata:
            p_test_cost, inputs, targets, outputs = self.pnn.Test(pdata[-5:])
            logging.info(
                'pnn test: cost: %s\n '
                'inputs:\n%s\n targets:\n%s\n outputs:\n%s\n delta:\n%s',
                p_test_cost, inputs, targets, outputs, outputs - targets)

        # summary
        action_summary[frame.action_id] += 1
        if step % 10000 == 0:
          logging.info('actions: %s', action_summary)
          action_summary = [0] * OUTPUT_DIM

        # reset on terminal
        if frame.terminal:
          logging.info(
              'task: %d steps: %9d episode length: %4d rewards: %6.2f',
              self.task_id, step, game.length, game.rewards)
          for f in list(game.last_frames)[-5:]:
            logging.info(
                '  galaxian:%s missile:%s bullets:%s incoming_enemies:%s',
                f.galaxian, f.missile, f.bullets, f.incoming_enemies)

          summary = tf.Summary()
          summary.value.add(
              tag='game/rewards', simple_value=float(game.rewards))
          summary.value.add(tag='game/length', simple_value=float(game.length))
          self.summary_writer.add_summary(summary, self.global_step.Eval())

          frame = game.Step('_')
          state = ac.InitialState()

  def plan(self, logits):
    self.cache = {}
    self.last_logits = logits
    frame = self.game.last_frames[-1]
    s = SFrame(frame)
    rewards, actions = self.search(frame.action, s, 0, 1.0)
    logging.info('search seq: %d gx: %3d my: %3d '
        'rewards: %7.2f actions: %-11s cache size: %d',
        frame.seq, frame.galaxian.x, frame.missile.y,
        rewards, actions, len(self.cache))
    return actions[0]

  def search(self, last_action, s, dep, decay):
    MAX_DEPTH = 6
    if dep >= MAX_DEPTH or s.rewards < 0:
      ret = s.rewards, ''  # TODO: + value from nn
    else:
      h = hash(s)  # TODO: hash collision?
      ret = self.cache.get(h)
      if ret is not None:
        return ret

      ret = (-10000, '')  # best rewards, action

      actions = ACTIONS[:]
      if s.missile.y < 200 or last_action in ['A', 'l', 'r']:
        actions = actions[:3]  # don't hold the A button
      if dep == 0:
        actions.sort(key=lambda a: self.last_logits[ACTION_ID[a]], reverse=1)
      else:
        random.shuffle(actions)

      threshold = 0
      if s.missile.y < 200:
        threshold = -1

      for action in actions:
        # TODO: simulate enemies and galaxian separately.
        t = self.game.Simulate(s, action, decay)
        tr, ta = self.search(action, t, dep+1, decay*0.99)
        if tr > ret[0]:
          ret = (tr, action+ta)
        if ret[0] > threshold:
          break  # don't be too greedy.

      self.cache[h] = ret
    logging.debug('%sgx: %d my: %d ret: %s',
        '  '*dep, s.galaxian.x, s.missile.y, ret)
    return ret


if __name__ == '__main__':
  tf.app.run()
