"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.tithx7juq
https://github.com/openai/universe-starter-agent/

TODO: Model based, Dyna, Sarsa, TD search, Monte Carlo.
TODO: Add paths into image?
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
import scipy.signal
import scipy.misc
import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool('debug', False, 'enable logging.debug')
flags.DEFINE_bool('log_steps', False, 'log steps')
flags.DEFINE_string('ui_tasks', '', 'tasks to start ui servers')
flags.DEFINE_string('eval_tasks', '', 'tasks to start servers in eval mode')
flags.DEFINE_string('ui_server', '../fceux-2.2.3-win32/fceux.exe -lua '
    'Z:\home\kelvinlau\galaxian\server.lua '
    'Z:\home\kelvinlau\galaxian\galaxian.nes',
    'ui server command')
flags.DEFINE_string('cc_server', './server ./galaxian.nes', 'cc server command')
flags.DEFINE_string('logdir', 'logs/2.33', 'Supervisor logdir')
flags.DEFINE_integer('port', 5001, 'server port to conenct')
flags.DEFINE_integer('num_workers', 1, 'num servers')
flags.DEFINE_bool('search', False, 'enable searching')
flags.DEFINE_bool('train', False, 'train or just play')
flags.DEFINE_bool('train_pnn', False, 'train pnn')
flags.DEFINE_bool('send_paths', False, 'send path to render by ui server')
flags.DEFINE_bool('send_value', False, 'send value to render by ui server')
flags.DEFINE_bool('verify_image', False, 'save image to verify input')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')


# Game constants.
NUM_STILL_ENEMIES = 10
NUM_INCOMING_ENEMIES = 7
WIDTH = 256
HEIGHT = 240
FRAME_SKIP = 5
GAP = 22
MISSILE_INIT_Y = 205
MISSILE_FRAME_SPEED = 4
MISSILE_STEP_SPEED = 20

DX = 4
HMAP_WIDTH = 256 / DX
FOCUS = 16
DATA_SIZE = 3 + (2*FOCUS+3) + NUM_INCOMING_ENEMIES*5 + 2*HMAP_WIDTH

PATH_LEN = 12

ACTIONS = ['_', 'L', 'R', 'A', 'l', 'r']
ACTION_ID = {ACTIONS[i]: i for i in xrange(len(ACTIONS))}
OUTPUT_DIM = len(ACTIONS)


def now():
  return int(time.time()*1e6)


class Timer:
  def __init__(self, name):
    self.name = name

  def __enter__(self):
    self.start = now()
    return self

  def __exit__(self, *args):
    self.end = now()
    self.interval = self.end - self.start
    print(self.name, self.interval, self.start, self.end)


def rehash(x, y):
  return (x*13 + y) % (2**64)


def iround(x):
  return x if isinstance(x, int) else round(x, 2)


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __hash__(self):
    return rehash(hash(self.x), hash(self.y))

  def __repr__(self):
    return '{},{}'.format(iround(self.x), iround(self.y))

  def __str__(self):
    return repr(self)

  def __add__(self, other):
    return Point(self.x+other.x, self.y+other.y)


class Rect(object):
  def __init__(self, o, w, h):
    self.x = o.x
    self.y = o.y
    self.w = w
    self.h = h

  @property
  def x1(self):
    return self.x - self.w

  @property
  def x2(self):
    return self.x + self.w

  @property
  def y1(self):
    return self.y

  @property
  def y2(self):
    return self.y + self.h - 1

  def __repr__(self):
    return '{}-{},{}-{}'.format(self.x1, self.x2, self.y1, self.y2)

  def __str__(self):
    return repr(self)

  def copy(self):
    return Rect(Point(self.x, self.y), self.w, self.h)


def intersected(a, b):
  tx = max(0, min(a.x2-b.x1+1, b.x2-a.x1+1))
  ty = max(0, min(a.y2-b.y1+1, b.y2-a.y1+1))
  return min(tx, ty, 4)


def one_hot(n, i):
  return [1 if j == i else 0 for j in xrange(n)]


def num_bits(mask):
  ret = 0
  while mask:
    if mask % 2:
      ret += 1
    mask = mask / 2
  return ret


def low_bit(mask):
  i = 0
  while mask % 2 == 0:
    mask /= 2
    i += 1
  return i


def sign(x):
  if x == 0:
    return 0
  return 1 if x > 0 else -1


# The nearest kx+b to y
def nearest_multiple(b, k, y):
  return int(round(1.0*(y-b)/k))*k+b


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

    self.score = self.NextInt()
    self.reward = math.sqrt(self.score/30.0)

    self.terminal = self.NextInt()

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

    # Penalty on holding the A button.
    # if prev_frames:
    #   pv = prev_frames[-1]
    #   if pv.missile.y < MISSILE_INIT_Y and self.action in 'Alr':
    #     self.reward -= 0.25

    self.galaxian = galaxian = self.NextPoint()

    self.missile = self.NextPoint()
    # penalty on miss
    # XXX: this may break if frame skip setting is change (currently 5).
    # if self.missile.y <= 4:
    #   self.reward -= 0.1

    # still enemies
    self.sdx = self.NextInt()
    self.masks = []
    for i in xrange(10):
      self.masks.append(self.NextInt())
    self.masks = self.masks[:NUM_STILL_ENEMIES]

    self.still_enemies = []
    for i, mask in enumerate(self.masks):
      x = self.sdx + 16 * i
      y = 105
      while mask:
        if mask % 2:
          self.still_enemies.append(Point(x, y))
        mask /= 2
        y -= 12
    assert len(self.still_enemies) <= 48, len(self.still_enemies)

    self.incoming_enemies = {}
    self.et = {}
    for i in xrange(self.NextInt()):
      eid = self.NextInt()
      e = self.NextPoint()
      row = self.NextInt()
      self.incoming_enemies[eid] = e
      self.et[eid] = enemy_type(row)
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

    self.data = []

    # missile x, y
    self.data.append(1. * (self.missile.x - galaxian.x) / WIDTH)
    self.data.append(1. * self.missile.y / MISSILE_INIT_Y
        if self.missile.y < MISSILE_INIT_Y else 0)
    logging.debug('missile %d,%d', self.missile.x, self.missile.y)

    # galaxian x
    self.data.append(1. * galaxian.x / WIDTH)

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
      ies.append([dx, dy] + one_hot(3, self.et[eid]))
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
      ee = e
      pe = None  # the furthest frame having this bullet
      steps = 0
      for pf in reversed(prev_frames):
        if eid in pf.bullets and pf.bullets[eid].y < ee.y:
          pe = pf.bullets[eid]
          steps += 1
        else:
          break
        ee = pe
      if pe:
        v = Point(1.*(e.x-pe.x)/steps, 1.*(e.y-pe.y)/steps)
      else:
        v = Point(0, 10)
      self.bv[eid] = v
      x1, x2, t = None, None, None
      if pe and pe.y < e.y < y1:
        x1 = int(round(1.*v.x/v.y*(y1-pe.y)+pe.x))
        x2 = int(round(1.*v.x/v.y*(y2-pe.y)+pe.x))
        if x1 > x2:
          x1, x2 = x2, x1
        t = 1.*(y1-e.y)/v.y*steps
      elif y1 <= e.y <= y2:
        x1 = x2 = e.x
        t = 0
      if x1 is not None:
        hit = max(0., 1.-t/12.)
        for i in xrange(ix(x1), ix(x2)+1):
          bmap[i] += hit

    self.data += imap
    self.data += bmap
    logging.debug('imap [%s]', hmap_string(imap))
    logging.debug('bmap [%s]', hmap_string(bmap))

    assert len(self.data) == DATA_SIZE, \
        '{} vs {}'.format(len(self.data), DATA_SIZE)
    self.data = np.array(self.data)

    self.image = np.zeros((WIDTH, HEIGHT))
    self.rect(galaxian, 7, 16)
    if self.missile.y < MISSILE_INIT_Y:
      self.rect(self.missile, 0, 4)
    for e in self.still_enemies:
      self.rect(e, 5, 11)
    for eid, e in self.incoming_enemies.iteritems():
      self.rect(e, 5, 11)
    for b in self.bullets.values():
      self.rect(b, 0, 4)
    self.image = np.reshape(self.image, [WIDTH, HEIGHT, 1])

  def NextToken(self):
    self._idx += 1
    return self._tokens[self._idx - 1]

  def NextInt(self):
    return int(self.NextToken())

  def NextPoint(self):
    return Point(self.NextInt(), self.NextInt())

  def rect(self, c, w, h, v=1.):
    x1 = max(c.x - w, 0)
    x2 = min(c.x + w + 1, WIDTH)
    y1 = max(c.y, 0)
    y2 = min(c.y + h, HEIGHT)
    if x1 >= x2 or y1 >= y2:
      return
    self.image[x1:x2, y1:y2] += np.full((x2-x1, y2-y1,), v)

  def CheckSum(self):
    return np.sum(self.data) + np.sum(self.image)


def is_dangerous(frame):
  for e in frame.incoming_enemies.values():
    if e.y > 150:
      return True
  for b in frame.bullets.values():
    if b.y > 150:
      return True
  return False


class SFrame:
  def __init__(self, src=None):
    if isinstance(src, Frame):
      self.galaxian = Rect(src.galaxian, 7, 16)
      self.missile = Rect(src.missile, 0, 4)
      self.sdx = src.sdx
      self.incoming_enemies = {i: Rect(e, 5, 11)
          for i, e in src.incoming_enemies.iteritems()}
      self.bullets = {i: Rect(e, 0, 4)
          for i, e in src.bullets.iteritems()}
      self.seq = src.seq
    else:
      assert isinstance(src, SFrame)
      self.galaxian = src.galaxian.copy()
      self.missile = src.missile.copy()
      self.sdx = src.sdx
      self.incoming_enemies = {
          i: e.copy() for i, e in src.incoming_enemies.iteritems()}
      self.bullets = {i: e.copy() for i, e in src.bullets.iteritems()}
      self.seq = src.seq
    self.rewards = 0
    self.b = None

  def __hash__(self):
    h = hash(self.seq)
    h = rehash(h, hash(self.galaxian.x))
    h = rehash(h, hash(self.missile))
    for eid in self.incoming_enemies:
      h = rehash(h, hash(eid))
    return h

  # TODO: __eq__

  def copy(self):
    return SFrame(self)


class Game:
  def __init__(self, port, pnn):
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect(('localhost', port))
    self._fin = self._sock.makefile()

    self._pnn = pnn
    self.last_frames = deque()
    self.length = 0
    self.rewards = 0
    self.scores = 0

    self.t_sim = 0

  def Start(self, seq=0, eval_mode=False):
    self._seq = seq
    self._sock.send('galaxian:start %d %d\n' % (seq+1, 1 if eval_mode else 0))
    assert self._fin.readline().strip() == 'ack'

  def Step(self, action, paths=[], info=''):
    if self.last_frames and self.last_frames[-1].terminal:
      self.last_frames.clear()
      self.length = 0
      self.rewards = 0
      self.scores = 0

    self._seq += 1

    msg = action + ' ' + str(self._seq)
    if paths:
      msg += ' paths ' + str(len(paths))
      for path in paths:
        for p in path:
          msg += ' ' + str(int(p.x))
          msg += ' ' + str(int(p.y))
    if info:
      msg += ' info ' + info
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
    self.scores += frame.score

    return frame

  def Simulate(self, s, action):
    frame = self.last_frames[-1]
    dep = s.seq - frame.seq
    decay = 0.99**dep

    start = now()

    t = s.copy()
    t.seq += 1

    # still enemies' speed is 1/5.
    t.sdx += frame.svx

    fired = 0
    # TODO: simulate 5 frames altogether?
    for i in xrange(1, FRAME_SKIP+1):
      if action in ['L', 'l']:
        t.galaxian.x = max(t.galaxian.x - 1, GAP)
      elif action in ['R', 'r']:
        t.galaxian.x = min(t.galaxian.x + 1, WIDTH-GAP)

      if t.missile.y < MISSILE_INIT_Y or fired:
        t.missile.y -= MISSILE_FRAME_SPEED
        if t.missile.y < 0:
          t.missile.y = MISSILE_INIT_Y
        fired = 0
      elif action in ['A', 'l', 'r'] and i == 1:
        fired = 1

      if t.missile.y >= MISSILE_INIT_Y:
        t.missile.x = t.galaxian.x

      for eid, e in t.bullets.iteritems():
        fe = frame.bullets[eid]
        v = frame.bv[eid]
        e.x = int(round(fe.x + v.x/FRAME_SKIP*(FRAME_SKIP*dep+i)))
        e.y = int(round(fe.y + v.y/FRAME_SKIP*(FRAME_SKIP*dep+i)))

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
        h = intersected(b, t.galaxian)
        if h:
          t.rewards -= 25 * h * decay
          t.b = b.copy()
          return t

      for e in t.incoming_enemies.values():
        h = intersected(e, t.galaxian)
        if h:
          t.rewards -= 20 * h * decay
          return t
      
      hits = 0
      if t.missile <= 117:
        for i, mask in enumerate(frame.masks):
          if mask > 0:
            e = Point(t.sdx + 16 * i, 105 - 12 * low_bit(mask))
            er = Rect(e, 5, 11)
            if intersected(e, t.missile) == 2:
              hits += 1
              t.rewards += 1
              break
        
      if t.missile.y < MISSILE_INIT_Y:
        for eid, e in t.incoming_enemies.items():
          if intersected(e, t.missile) == 2:
            t.rewards += (frame.et[eid] + 2)
            del t.incoming_enemies[eid]
            hits += 1

      if hits:
        t.missile.y = MISSILE_INIT_Y
        t.missile.x = t.galaxian.x

    self.t_sim += now()-start

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


def const_var(name, shape, value=0.):
  return tf.get_variable(name, shape,
                         initializer=tf.constant_initializer(value))


def linear(x, n, name, w_std=1.):
  with tf.variable_scope(name):
    m = x.get_shape().as_list()[1]
    w_init = np.random.randn(m, n).astype(np.float32)
    w_init *= w_std / np.sqrt(np.square(w_init).sum(axis=0, keepdims=1))
    w = tf.Variable(tf.constant(w_init), name='w')
    b = const_var('b', [n], 0.)
    return tf.matmul(x, w) + b


def conv2d(x, num_filters, name, filter_size, stride):
  with tf.variable_scope(name):
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]),
        num_filters]

    fan_in = np.prod(filter_shape[:3])
    fan_out = np.prod(filter_shape[:2]) * num_filters
    wb = np.sqrt(6. / (fan_in + fan_out))

    w = tf.get_variable('w', filter_shape,
                        initializer=tf.random_uniform_initializer(-wb, wb))
    b = const_var('b', [1, 1, 1, num_filters], value=0.)
    return tf.nn.conv2d(x, w, stride_shape, 'VALID') + b


def flatten(x):
  return tf.reshape(x, [-1, np.prod(x.get_shape().as_list()[1:])])


class ACNeuralNetwork:
  def __init__(self, name, global_ac=None):
    with tf.variable_scope(name):
      # Input.
      x = self.image = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, 1],
          name='image')
      for i in xrange(5):
        x = tf.nn.elu(conv2d(x, 2**(i+2), 'c'+str(i), [3, 3], [2, 2]))
      x = flatten(x)
      self.data = tf.placeholder(tf.float32, [None, DATA_SIZE],
          name='data')
      x = tf.concat(axis=1, values=[x, self.data])

      # Make batch size as time dimension.
      x = tf.expand_dims(x, [0])

      # LSTM.
      LSTM_SIZE = 256
      lstm = rnn.BasicLSTMCell(LSTM_SIZE)

      c_init = np.zeros((1, lstm.state_size.c), np.float32)
      h_init = np.zeros((1, lstm.state_size.h), np.float32)
      self.state_init = [c_init, h_init]

      c_in = tf.placeholder(tf.float32, [1, lstm.state_size.c], name='c')
      h_in = tf.placeholder(tf.float32, [1, lstm.state_size.h], name='h')
      self.state_in = [c_in, h_in]

      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
          lstm, x, initial_state=rnn.LSTMStateTuple(c_in, h_in),
          sequence_length=tf.shape(self.image)[:1])
      lstm_c, lstm_h = lstm_state
      self.state_out = [lstm_c, lstm_h]

      # 1 more fully connected layer.
      x = tf.reshape(lstm_outputs, [-1, LSTM_SIZE])
      x = tf.nn.elu(linear(x, 128, 'fc0'))

      # Output logits and value.
      self.logits = linear(x, OUTPUT_DIM, 'logits', w_std=0.01)
      self.value = tf.reshape(linear(x, 1, 'value'), [-1])
      self.action = categorical_sample(self.logits, OUTPUT_DIM)

      self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
          scope=name)
      assert len(self.var_list) == 18, len(self.var_list)

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
        self.loss = policy_loss + 0.5 * value_loss - 0.05 * entropy

        grads = tf.gradients(self.loss, self.var_list)
        grads_clipped, _ = tf.clip_by_global_norm(grads, 10.0)

        self.optimizer = \
            tf.train.AdamOptimizer(FLAGS.learning_rate).apply_gradients(
                zip(grads_clipped, global_ac.var_list))

        self.sync = tf.group(*[dst.assign(src)
          for dst, src in zip(self.var_list, global_ac.var_list)])

        batch_size = tf.to_float(tf.shape(self.image)[0])
        summaries = [
          tf.summary.image("image", tf.transpose(self.image, [0, 2, 1, 3])),
          tf.summary.scalar("policy_loss", policy_loss / batch_size),
          tf.summary.scalar("value_loss", value_loss / batch_size),
          tf.summary.scalar("entropy", entropy / batch_size),
          tf.summary.scalar("loss", self.loss / batch_size),
          tf.summary.scalar("grad_global_norm", tf.global_norm(grads)),
          tf.summary.scalar("var_global_norm",
              tf.global_norm(self.var_list)),
        ]
        for v in self.var_list:
          hist_name = "hist/"+v.name[len(name)+1:-2]
          summaries.append(tf.summary.histogram(hist_name, v))
        self.summary_op = tf.summary.merge(summaries)

  def InitialState(self):
    return self.state_init

  def Eval(self, frame, state):
    ret = tf.get_default_session().run(
        [self.action, self.value] + self.state_out + [self.logits], {
            self.data: [frame.data],
            self.image: [frame.image],
            self.state_in[0]: state[0],
            self.state_in[1]: state[1],
        })
    return ret[0][0], ret[1][0], ret[2:4], ret[4][0]

  def Train(self, experience, return_summary=False):
    GAMMA = 0.95
    LAMBDA = 1.0

    _, last_frame, _, last_state = experience[-1]
    terminal = last_frame.terminal
    last_value = self.Eval(last_frame, last_state)[1] if not terminal else 0.
    data = np.array([f.data for f, _, _, _ in experience])
    images = np.array([f.image for f, _, _, _ in experience])
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
      self.data: data,
      self.image: images,
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
    with tf.variable_scope(name):
      INPUT_SIZE = 6
      OUTPUT_SIZE = 2*PATH_LEN
      LSTM_SIZE = 16
      self.input = tf.placeholder(tf.float32, [None, PATH_LEN, INPUT_SIZE])
      self.keep_prob = tf.placeholder(tf.float32)
      lstm0 = rnn.BasicLSTMCell(LSTM_SIZE)
      lstm1 = rnn.BasicLSTMCell(LSTM_SIZE)
      lstm1 = rnn.DropoutWrapper(lstm1, output_keep_prob=self.keep_prob)
      lstm = rnn.MultiRNNCell([lstm0, lstm1])
      rnn_out, state = tf.nn.dynamic_rnn(lstm, self.input, dtype=tf.float32)
      rnn_out = tf.transpose(rnn_out, [1, 0, 2])
      rnn_out = rnn_out[-1]
      self.output = linear(rnn_out, OUTPUT_SIZE, name='output')

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
      coor += one_hot(3, f.et[eid])
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

  logging.info('variables:')
  for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    logging.info('  %s: %s', var.name, var.get_shape())

  ui_tasks = parse_ints(FLAGS.ui_tasks)
  eval_tasks = parse_ints(FLAGS.eval_tasks)
  MAX_WORKERS = 16
  workers = [Worker(global_step, global_ac, pnn, summary_writer, i,
                    i in ui_tasks, i in eval_tasks)
      for i in xrange(MAX_WORKERS)]

  sv = tf.train.Supervisor(logdir=FLAGS.logdir,
                           global_step=global_step.var,
                           saver=tf.train.Saver(
                               max_to_keep=10,
                               keep_checkpoint_every_n_hours=1,
                               pad_step_number=True),
                           summary_op=None,
                           summary_writer=summary_writer,
                           save_model_secs=600,
                           save_summaries_secs=60)

  config = tf.ConfigProto(
      intra_op_parallelism_threads=FLAGS.num_workers,
      log_device_placement=False,
      gpu_options=tf.GPUOptions(allow_growth=True))

  with sv.managed_session(config=config) as sess, sess.as_default():
    logging.info('ac: %s', format_list(global_ac.CheckSum()))
    logging.info('pnn: %s', format_list(pnn.CheckSum()))

    for worker in workers[:FLAGS.num_workers]:
      worker.begin(sv, sess)

    while any(w.is_alive() for w in workers):
      time.sleep(1)


class Worker(threading.Thread):
  def __init__(self, global_step, global_ac, pnn, summary_writer, task_id,
               is_ui_server, eval_mode):
    threading.Thread.__init__(self, name='worker-'+str(task_id))
    self.daemon = True

    self.global_step = global_step
    self.pnn = pnn
    self.summary_writer = summary_writer
    self.task_id = task_id
    self.is_ui_server = is_ui_server
    self.eval_mode = eval_mode
    self.port = FLAGS.port + task_id
    self.ac = ACNeuralNetwork('ac-' + str(task_id), global_ac=global_ac)

    self.last_actions = None

  def begin(self, sv, sess):
    port = self.port
    if self.is_ui_server:
      cmd = FLAGS.ui_server.split(' ')
      assert port == 5001, port
    else:
      cmd = FLAGS.cc_server.split(' ') + [str(port)]
    logging.info('running %s', ' '.join(cmd))
    self.server = subprocess.Popen(cmd,
            stdout=open('/tmp/galaxian-{}.stdout'.format(self.task_id), 'w'),
            stderr=open('/tmp/galaxian-{}.stderr'.format(self.task_id), 'w'))

    self.sv = sv
    self.sess = sess

    self.start()

  def run(self):
    time.sleep(10)  # wait for server startup
    game = self.game = Game(self.port, self.pnn)
    game.Start(eval_mode=self.eval_mode)
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

    send_paths = FLAGS.send_paths and self.is_ui_server
    send_value = FLAGS.send_value and self.is_ui_server

    sess = self.sess
    with sess.as_default():
      ac.Sync()

      while not self.sv.should_stop():
        start = now()

        # eval action
        action_prob, value, state, logits = ac.Eval(frame, state)
        nn_action = ACTIONS[action_prob.argmax()]
        eval_dur = now() - start
        action = self.search(nn_action, logits)
        paths = frame.paths.values() if send_paths else []

        # take action
        info = ''
        if send_value:
          info = 'v=' + str(value)
        if self.is_ui_server and nn_action != action:
          info += ' o'
        frame1 = game.Step(action, paths=paths, info=info)
        experience.append((frame, frame1, value, state))
        frame = frame1

        step += 1
        step_dur = now() - start
        logging.log(lvl,
            "Step %d value: %7.3f logits: %s action: %s reward: %5.2f "
            "eval_dur: %d step_dur: %d",
            step, value, format_list(logits, fmt='%7.3f'), frame.action,
            frame.reward, eval_dur, step_dur)

        if FLAGS.verify_image and step % 100 == 0 and self.task_id == 0:
          scipy.misc.imsave('image.jpg',
              np.transpose(np.reshape(frame.data, (WIDTH, HEIGHT))))

        # policy training
        TRAIN_LENGTH = 20
        if FLAGS.train and (len(experience) >= TRAIN_LENGTH or frame.terminal):
          trainings += 1
          do_summary = trainings % 10 == 0

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
          MINI_BATCH_SIZE = 32
          EPOCHS = 2
          if len(pdata) >= 10000:
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
            summary = tf.Summary()
            summary.value.add(tag='pnn/train_cost', simple_value=p_train_cost)
            self.summary_writer.add_summary(summary, self.global_step.Eval())

          # testing
          if step % 10000 == 0 and pdata:
            p_test_cost, inputs, targets, outputs = self.pnn.Test(pdata[-5:])
            logging.info('pnn test cost: %s', p_test_cost)
            logging.debug('pnn test inputs:\n%s\n targets:\n%s\n '
                'outputs:\n%s\n delta:\n%s',
                inputs, targets, outputs, outputs - targets)
            summary = tf.Summary()
            summary.value.add(tag='pnn/test_cost', simple_value=p_test_cost)
            self.summary_writer.add_summary(summary, self.global_step.Eval())

        # summary
        action_summary[frame.action_id] += 1
        if step % 10000 == 0:
          logging.info('actions: %s', zip(ACTIONS, action_summary))
          action_summary = [0] * OUTPUT_DIM

        # reset on terminal
        if frame.terminal:
          logging.info('task: %d steps: %9d episode length: %4d rewards: %6.2f '
              'scores: %5d',
              self.task_id, step, game.length, game.rewards, game.scores)
          if FLAGS.search and not FLAGS.train:
            for f in list(game.last_frames)[-10:]:
              logging.info(
                  '  galaxian:%s missile:%s bullets:%s bv:%s '
                  'incoming_enemies:%s',
                  f.galaxian, f.missile, f.bullets, f.bv, f.incoming_enemies)

          prefix = 'game-{}/'.format(self.task_id)
          summary = tf.Summary()
          summary.value.add(tag=prefix+'rewards', simple_value=game.rewards)
          summary.value.add(tag=prefix+'scores', simple_value=game.scores)
          summary.value.add(tag=prefix+'length', simple_value=game.length)
          self.summary_writer.add_summary(summary, self.global_step.Eval())

          frame = game.Step('_')
          state = ac.InitialState()

  MAX_DEPTH = 8

  def search(self, nn_action, logits):
    start = now()
    self.game.t_sim = 0

    frame = self.frame = self.game.last_frames[-1]

    if not (FLAGS.search and is_dangerous(frame)):
      self.last_actions = None
      return nn_action

    logits = logits.copy()
    bonus_actions = ''
    if frame.galaxian.x < 50:
      bonus_actions += 'Rr'
    elif frame.galaxian.x > 256-50:
      bonus_actions += 'Ll'
    for a in bonus_actions:
      logits[ACTION_ID[a]] += 1.0
    logits -= np.max(logits)
    self.logits = logits

    s = SFrame(frame)
    self.cache = {}

    rewards, actions, bb = self.dfs(frame.action, s, forward_actions=nn_action)
    stra = 'n'

    if self.last_actions and rewards < 0:
      r, a, b = self.dfs(frame.action, s, forward_actions=self.last_actions[1:])
      if r > rewards:
        rewards, actions, bb = r, a, b
        stra = 'c'

    if rewards < 0:
      r, a, b = self.dfs(frame.action, s)
      if r > rewards:
        rewards, actions, bb = r, a, b
        stra = 's'

    # TODO: do this if missile if fired too.
    groutes = None
    grewards = None
    if False and s.missile.y >= MISSILE_INIT_Y and rewards <= 1:
      groutes = []
      grewards = []
      for eid, e in s.incoming_enemies.iteritems():
        path = frame.paths.get(eid)
        if path is None:
          continue
        prev = e
        for t in xrange(1, Worker.MAX_DEPTH+1):
          pos = path[t-1]
          pos = Rect(pos, 5, 11)
          MISSILE_MIN_HIT = MISSILE_INIT_Y + (FRAME_SKIP-1) * \
              MISSILE_FRAME_SPEED
          # TODO: Count hits between hits too.
          hit = Point(
              nearest_multiple(s.missile.x, FRAME_SKIP, pos.x),
              nearest_multiple(MISSILE_MIN_HIT, MISSILE_STEP_SPEED, pos.y))
          hit = Rect(hit, 0, 4)
          if hit.y > MISSILE_MIN_HIT or not intersected(hit, pos):
            continue
          ty = (MISSILE_INIT_Y+MISSILE_FRAME_SPEED-hit.y)/MISSILE_STEP_SPEED
          tx = (hit.x-frame.galaxian.x)/FRAME_SKIP
          ts = t - abs(tx) - ty
          if ts < 0:
            continue
          # TODO: More accurate difficulty.
          difficulty = ty + abs(pos.x-prev.x)
          forward_actions = ''
          for a, t in [('R' if tx > 0 else 'L', abs(tx)), ('_', ts)]:
            for i in xrange(t):
              forward_actions += a
          groutes.append((difficulty, forward_actions))
          prev = pos
      groutes.sort()
      groutes = groutes[:4]
      for i, (_, forward_actions) in enumerate(groutes):
        if rewards > 1:
          break
        r, a = self.dfs(frame.action, s, forward_actions=forward_actions,
                        next_candidates='Alr')
        grewards.append(r)
        if r >= rewards:
          rewards, actions = r, a
          stra = 'g'+str(i)

    end = start + 8000
    for i in xrange(10):
      if rewards > 0 or now() > end:
        break
      r, a = self.mcts(s)
      if r > rewards:
        rewards, actions = r, a
        stra = 'm'

    if 0 and rewards < 0:
      ret = nn_action
      stra = 'n'
      self.last_actions = None
    else:
      ret = actions[0]
      self.last_actions = actions

    if not FLAGS.train:
      dur = now()-start
      logging.info(
          'search seq: %d gx: %3d my: %3d '
          'stra: %-2s rewards: %7.2f %1s nn_action: %s '
          'actions: %-11s cache size: %5d groutes: %s grewards: %s '
          'dur: %d t_sim: %d bb: %s',
          frame.seq, frame.galaxian.x, frame.missile.y,
          stra, rewards, 'o' if nn_action != ret else '', nn_action,
          actions, len(self.cache), groutes, grewards,
          dur, self.game.t_sim, bb)
    return ret

  def dfs(self, last_action, s, forward_actions=None, next_candidates=None):
    dep = s.seq - self.frame.seq
    if dep >= Worker.MAX_DEPTH or s.rewards < 0:
      ret = s.rewards, '', s.b  # TODO: + value from nn
    else:
      #h = hash(s)  # TODO: hash collision?
      ret = self.cache.get(s)
      if ret is not None:
        return ret

      ret = (-10000, '')  # best rewards, actions

      candidates = None

      limited = 0
      if forward_actions:
        candidates = [forward_actions[0]]
        forward_actions = forward_actions[1:]
        limited = 1
      elif next_candidates:
        candidates = next_candidates
        next_candidates = None
        limited = 1

      if candidates is None:
        candidates = ACTIONS[:]
        if s.missile.y < MISSILE_INIT_Y or last_action in 'Alr':
          candidates = candidates[:3]  # don't hold the A button
        # TODO: More likehood to choose last_action.
        if dep == 0:
          prob = [math.exp(logit) * random.random() for logit in self.logits]
          candidates.sort(key=lambda a: prob[ACTION_ID[a]], reverse=1)
        else:
          random.shuffle(candidates)

      for action in candidates:
        # TODO: simulate enemies and galaxian separately.
        t = self.game.Simulate(s, action)
        tr, ta, tb = self.dfs(action, t, forward_actions=forward_actions,
                              next_candidates=next_candidates)
        if tr > ret[0]:
          ret = (tr, action+ta, tb)
        if ret[0] >= 0:
          break  # just to survive

      if not limited:
        self.cache[s] = ret
    return ret

  def mcts(self, s):
    actions = ''
    last_action = self.frame.action
    for dep in xrange(0, Worker.MAX_DEPTH):
      candidates = ACTIONS
      logits = self.logits
      if s.missile.y < MISSILE_INIT_Y or last_action in 'Alr':
        candidates = candidates[:3]  # don't hold the A button
        logits = logits[:3]
      if dep == 0:
        probs = np.exp(logits)
        probs = np.maximum(probs, .01)
        probs /= np.sum(probs)
        idx = np.where(np.random.multinomial(1, probs))[0][0]
        action = candidates[idx]
      else:
        action = random.choice(candidates)

      actions += action
      last_action = action
      s = self.game.Simulate(s, action)
      if s.rewards < 0:
        return s.rewards, actions
    return s.rewards, actions


if __name__ == '__main__':
  tf.app.run()
