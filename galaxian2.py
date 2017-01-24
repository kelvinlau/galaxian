"""Galaxian deep neural network.

Ref:
https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df#.tithx7juq

TODO: Save png to verify input data.
TODO: Training the model only on score increases?
TODO: 2D sensors.
TODO: Enemy type.
TODO: Use a small separated nn for incoming enemy curves.
TODO: In-bound but edge tiles should have some penality?
TODO: Save on dangerous situations.
TODO: Fewer layer.
TODO: LSTM.
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
#import cv2
import numpy as np
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('server', '', 'server binary')
flags.DEFINE_string('rom', './galaxian.nes', 'galaxian nes rom file')
flags.DEFINE_float('eps', None, 'initial epsilon')
flags.DEFINE_string('checkpoint_dir', 'galaxian2w/', 'Checkpoint dir')
flags.DEFINE_integer('port', 62343, 'server port to conenct')


# Game input/output.
NUM_STILL_ENEMIES = 10
NUM_INCOMING_ENEMIES = 6
RAW_IMAGE = False
if RAW_IMAGE:
  NUM_SNAPSHOTS = 4
  SCALE = 2
  WIDTH = 256/SCALE
  HEIGHT = 240/SCALE
  SIDE = 84
else:
  DX = 8
  WIDTH = 256 / DX
  FOCUS = 16
  NUM_SNAPSHOTS = 5
  NIE = 10
  INPUT_DIM = 4 + (2*FOCUS+3) + NIE + 2*WIDTH + (2*FOCUS)


ACTION_NAMES = ['_', 'L', 'R', 'A', 'l', 'r']
ACTION_ID = {ACTION_NAMES[i]: i for i in xrange(len(ACTION_NAMES))}
OUTPUT_DIM = len(ACTION_NAMES)

# Hyperparameters.
DOUBLE_Q = True
GAMMA = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05 if DOUBLE_Q else 0.1
EXPLORE_STEPS = 2000000
OBSERVE_STEPS = 5000
REPLAY_MEMORY = 100000 if not RAW_IMAGE else 2000  # 2000 = ~6G memory
MINI_BATCH_SIZE = 32
TRAIN_INTERVAL = 1
UPDATE_TARGET_NETWORK_INTERVAL = 10000

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
  def __init__(self, line, prev_frames):
    """Parse a Frame from a line."""
    self._tokens = line.split()
    self._idx = 0

    self.seq = self.NextInt()

    # reward is sqrt of score/30 capped under 3.
    self.reward = min(3, math.sqrt(self.NextInt()/30.0))

    self.terminal = self.NextInt()

    self.action = self.NextToken()
    self.action_id = ACTION_ID[self.action]

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
      self.incoming_enemies[eid] = self.NextPoint()

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
      #print('fired', fired, 'missile', self.missile.y)

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
          num = NumBits(mask)
          for x in xrange(max(ex-4, x1), min(ex+4, x2)):
            smap[x-x1] += num / 7.
          if ex < x1:
            sl = 1
          if ex >= x2:
            sr = 1
      svx = 0
      if prev_frames:
        svx = Sign(self.sdx - prev_frames[-1].sdx)
      self.data += smap
      self.data.append(sl)
      self.data.append(sr)
      self.data.append(svx)
      #print('smap [', ''.join(['x' if h > 0 else '_' for h in smap]), ']',
      #      sl, sr, svx)

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
        return max(0, min(2*WIDTH-1, (x-galaxian.x+256)/DX))
      # out-of-bound tiles have penality.
      hmap = [0. if ix(0) <= i <= ix(255) else 1. for i in range(WIDTH*2)]
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
      self.data += fmap
      #print('hmap [', ''.join(['x' if h > 0 else '_' for h in hmap]), ']')
      #print('fmap [', ''.join(['x' if h > 0 else '_' for h in fmap]), ']')

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

  @staticmethod
  def InvertX(dx):
    if dx > 0:
      return (256 - dx) / 256.0
    elif dx < 0:
      return (-256 - dx) / 256.0
    else:
      return 1

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
    self._seq = 0
    self._prev_frames = deque()

  def Start(self):
    self._sock.send('galaxian:start\n')
    assert self._fin.readline().strip() == 'ack'

  def Step(self, action):
    # Don't hold the A button.
    if self._prev_frames:
      pv = self._prev_frames[-1]
      if pv.missile.y >= 200 and pv.action in ['A', 'l', 'r']:
        if action == 'A':
          action = '_'
        elif action == 'l':
          action = 'L'
        elif action == 'r':
          action = 'R'

    self._seq += 1
    #print()
    #print(action, self._seq)

    self._sock.send(action + ' ' + str(self._seq) + '\n')

    line = self._fin.readline().strip()

    frame = Frame(line, self._prev_frames)

    assert frame.seq == self._seq, 'Expecting %d, got %d' % (self._seq,
        frame.seq)

    if frame.terminal:
      self._prev_frames.clear()
    else:
      self._prev_frames.append(frame)
      if len(self._prev_frames) > NUM_SNAPSHOTS-1:
        self._prev_frames.popleft()

    return frame


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
  return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class NeuralNetwork:
  def __init__(self, name, trainable=True):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02), trainable=trainable)

    with tf.variable_scope(name):
      self.keep_prob = tf.placeholder(tf.float32)

      if not RAW_IMAGE:
        # Input 1.
        self.input1 = tf.placeholder(tf.float32, [None, INPUT_DIM-NIE])
        print('input1:', self.input1.get_shape())

        # Input 2.
        self.ies = tf.placeholder(tf.float32,
                                  [None, NUM_INCOMING_ENEMIES, 2*NUM_SNAPSHOTS])
        ies0 = tf.reshape(self.ies, [-1, 2*NUM_SNAPSHOTS])
        NIE1 = 8
        ies1 = tf.nn.relu(tf.matmul(ies0, var([2*NUM_SNAPSHOTS, NIE1])) +
                          var([NIE1]))
        ies1 = tf.nn.dropout(ies1, self.keep_prob)
        print('ies1', ies1.get_shape())
        ies2 = tf.nn.relu(tf.matmul(ies1, var([NIE1, NIE])) + var([NIE]))
        ies2 = tf.nn.dropout(ies2, self.keep_prob)
        print('ies2', ies2.get_shape())
        ies3 = tf.reshape(ies2, [-1, NUM_INCOMING_ENEMIES, NIE])
        print('ies3', ies3.get_shape())
        input2 = tf.reduce_sum(ies3, axis = 1)
        print('input2', input2.get_shape())

        self.input = tf.concat_v2([self.input1, input2], axis=1)
        print('input', self.input.get_shape())

        N1 = 32
        N2 = 24
        N3 = 16

        fc1 = tf.nn.relu(tf.matmul(self.input, var([INPUT_DIM, N1])) +
                         var([N1]))
        fc1 = tf.nn.dropout(fc1, self.keep_prob)

        fc2 = tf.nn.relu(tf.matmul(fc1, var([N1, N2])) + var([N2]))
        fc2 = tf.nn.dropout(fc2, self.keep_prob)

        fc3 = tf.nn.relu(tf.matmul(fc2, var([N2, N3])) + var([N3]))
        fc3 = tf.nn.dropout(fc3, self.keep_prob)

        value = tf.matmul(fc3, var([N3, 1])) + var([1])
        advantage = tf.matmul(fc3, var([N3, OUTPUT_DIM])) + var([OUTPUT_DIM])
        # Simple dueling.
        # TODO: Max dueling or avg dueling.
        self.output = advantage + value
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
    assert len(self.theta) == (10 if RAW_IMAGE else 14), len(self.theta)

    if trainable:
      # Training.
      self.action = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
      self.y = tf.placeholder(tf.float32, [None])
      q_action = tf.reduce_sum(tf.multiply(self.output, self.action),
          reduction_indices = 1)
      self.cost = tf.reduce_mean(ClippedError(q_action - self.y))
      self.optimizer = tf.train.RMSPropOptimizer(
          learning_rate=0.00025, momentum=.95, epsilon=1e-2).minimize(self.cost)

  def Vars(self):
    return self.theta

  def Eval(self, frames, keep_prob):
    return self.output.eval(feed_dict = {
        self.input1: [f.datax for f in frames],
        self.ies: [f.ies for f in frames],
        self.keep_prob: keep_prob,
    })

  def Train(self, tnn, mini_batch, keep_prob):
    frame_batch = [d[0] for d in mini_batch]
    action_batch = [d[1] for d in mini_batch]
    frame1_batch = [d[2] for d in mini_batch]

    t_q1_batch = tnn.Eval(frame1_batch, keep_prob)
    y_batch = [0] * len(mini_batch)
    if not DOUBLE_Q:
      for i in xrange(len(mini_batch)):
        reward = frame1_batch[i].reward
        if frame1_batch[i].terminal:
          y_batch[i] = reward
        else:
          y_batch[i] = reward + GAMMA * np.max(t_q1_batch[i])
    else:
      q1_batch = self.Eval(frame1_batch, keep_prob)
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
        self.keep_prob: keep_prob,
    }
    self.optimizer.run(feed_dict = feed_dict)
    return self.cost.eval(feed_dict = feed_dict), y_batch[-1]

  def CopyFrom(self, sess, src):
    for v1, v2 in zip(self.Vars(), src.Vars()):
      sess.run(v1.assign(v2))

  def CheckSum(self):
    return [np.sum(var.eval()) for var in self.Vars()]


def FormatList(l):
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


def main(unused_argv):
  port = FLAGS.port
  if FLAGS.server:
    server = subprocess.Popen([FLAGS.server, FLAGS.rom, str(port)])
    time.sleep(1)

  memory = deque()
  memoryx = deque()
  nn = NeuralNetwork('nn')
  tnn = NeuralNetwork('tnn', trainable=False)
  game = Game(port)
  game.Start()
  frame = game.Step('_')

  saved_step = SavedVar(0, 'step')
  saved_epsilon = SavedVar(INITIAL_EPSILON, 'epsilon')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(nn.Vars() + [saved_step.var, saved_epsilon.var])
    checkpoint_dir = FLAGS.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print("Restored from", ckpt.model_checkpoint_path)
    else:
      print("No checkpoint found")

    tnn.CopyFrom(sess, nn)

    step = saved_step.Eval()
    epsilon = FLAGS.eps or saved_epsilon.Eval()
    cost = 1e9
    y_val = 1e9
    while True:
      keep_prob = 1 - epsilon + FINAL_EPSILON

      q_val = nn.Eval([frame], keep_prob)[0]
      action = ACTION_NAMES[np.argmax(q_val)]

      frame1 = game.Step(action)

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

      if step % TRAIN_INTERVAL == 0 and step > OBSERVE_STEPS:
        mini_batch = random.sample(memory, min(len(memory), MINI_BATCH_SIZE))
        mini_batch += random.sample(memoryx, min(len(memoryx), MINI_BATCH_SIZE))
        mini_batch.append(memory[-1])
        cost, y_val = nn.Train(tnn, mini_batch, keep_prob)

      frame = frame1
      step += 1

      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if step % UPDATE_TARGET_NETWORK_INTERVAL == 0:
        print('Target network before:', tnn.CheckSum())
        tnn.CopyFrom(sess, nn)
        print('Target network after:', tnn.CheckSum())

      if step % SAVE_INTERVAL == 0:
        saved_step.Assign(sess, step)
        saved_epsilon.Assign(sess, epsilon)
        save_path = saver.save(sess, checkpoint_dir + CHECKPOINT_FILE,
                               global_step = step)
        print("Saved to", save_path)

      print("Step %d keep: %.6f nn: %s q: %-49s action: %s reward: %5.2f "
          "cost: %8.3f y: %8.3f" %
          (step, keep_prob, FormatList(nn.CheckSum()), FormatList(q_val),
            frame1.action, frame1.reward, cost, y_val))


if __name__ == '__main__':
  tf.app.run()
