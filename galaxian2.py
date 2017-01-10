"""Galaxian NN trainer.

Galaxian deep neural network.

Ref: https://www.nervanasys.com/demystifying-deep-reinforcement-learning/
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


WIDTH = 96
HEIGHT = 32
NUM_SNAPSHOTS = 3
INPUT_DIM = NUM_SNAPSHOTS * (WIDTH + WIDTH*HEIGHT*2) + 2
OUTPUT_NAMES = ['stay', 'left', 'right', 'fire']
OUTPUT_IDS = {'stay': 0, 'left': 1, 'right': 2, 'fire': 3}
OUTPUT_DIM = len(OUTPUT_NAMES)  # TODO(kelvinlau): 6?


class Frame:
  def __init__(self):
    self.s = np.zeros((WIDTH, NUM_SNAPSHOTS))
    self.i = np.zeros((WIDTH, HEIGHT, NUM_SNAPSHOTS))
    self.b = np.zeros((WIDTH, HEIGHT, NUM_SNAPSHOTS))
    self.x = 0.0
    self.m = 0.0
    self.action = None
    self.action_id = None


class NeuralNetwork:
  def __init__(self):
    # Still enemies.
    self.input_s = tf.placeholder(tf.float32,
        [None, WIDTH, NUM_SNAPSHOTS])

    # Incoming enemies.
    self.input_i = tf.placeholder(tf.float32,
        [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])

    # Incoming bullets.
    self.input_b = tf.placeholder(tf.float32,
        [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])

    # galaxian_x
    self.input_x = tf.placeholder(tf.float32, [None, 1])

    # missile_y.
    self.input_m = tf.placeholder(tf.float32, [None, 1])

    # 96 x 32 x 6
    input_ib = tf.concat(3, [self.input_i, self.input_b])

    zeros = lambda shape: tf.Variable(tf.zeros(shape))

    # 23 x 15 x 32
    conv1 = tf.nn.relu(tf.nn.conv2d(
      input_ib, zeros([8, 4, 6, 32]), strides = [1, 4, 2, 1],
      padding = "VALID")
      + zeros([32]))

    # 10 x 6 x 64
    conv2 = tf.nn.relu(tf.nn.conv2d(
      conv1, zeros([4, 4, 32, 64]), strides = [1, 2, 2, 1],
      padding = "VALID")
      + zeros([64]))

    # 9 x 5 x 64
    conv3 = tf.nn.relu(tf.nn.conv2d(
      conv2, zeros([2, 2, 64, 64]), strides = [1, 1, 1, 1],
      padding = "VALID")
      + zeros([64]))

    # 2880 + 288 + 1 + 1 = 3170
    # TODO: Conv input_s too?
    all_flat = tf.concat(1, [
      tf.reshape(conv3, [-1, 2880]),
      tf.reshape(self.input_s, [-1, WIDTH*NUM_SNAPSHOTS]),
      self.input_x,
      self.input_m])

    fc4 = tf.nn.relu(tf.matmul(all_flat, zeros([3170, 784])) + zeros([784]))

    # Neural Network output.
    self.output_layer = (tf.matmul(fc4, zeros([784, OUTPUT_DIM])) +
        zeros([OUTPUT_DIM]))

    # NN training.
    self.action = tf.placeholder(tf.float32, [None, OUTPUT_DIM])
    self.q_target = tf.placeholder(tf.float32, [None])
    q_action = tf.reduce_sum(tf.mul(self.output_layer, self.action),
        reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(q_action - self.q_target))
    self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)

  def FeedDict(self, frames):
    return {
      self.input_s: [f.s for f in frames],
      self.input_i: [f.i for f in frames],
      self.input_b: [f.b for f in frames],
      self.input_x: [[f.x] for f in frames],
      self.input_m: [[f.m] for f in frames],
    }

  def Eval(self, frames):
    return self.output_layer.eval(feed_dict = self.FeedDict(frames))

  def Train(self, mini_batch):
    frame_batch = [d[0] for d in mini_batch]
    action_batch = [d[1] for d in mini_batch]
    reward_batch = [d[2] for d in mini_batch]
    frame1_batch = [d[3] for d in mini_batch]

    q1_batch = self.Eval(frame1_batch)
    q_target_batch = [
        reward_batch[i] if reward_batch[i] < 0 else
        reward_batch[i] + LEARNING_RATE * np.max(q1_batch[i])
        for i in xrange(len(mini_batch))
    ]

    feed_dict = self.FeedDict(frame_batch)
    feed_dict[self.action] = action_batch
    feed_dict[self.q_target] = q_target_batch
    self.optimizer.run(feed_dict = feed_dict)


class Point:
  def __init__(self, x, y):
    self.x = x
    self.y = y


class Timer:
  def __init__(self, name):
    self.name = name

  def __enter__(self):
    self.start = time.time()
    return self

  def __exit__(self, *args):
    self.end = time.time()
    self.interval = self.end - self.start
    print(self.name, self.interval * 1e6)


class Game:
  def __init__(self):
    self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._sock.connect(('localhost', 62343))
    self._fin = self._sock.makefile()
    self._seq = 0

  def Step(self, action):
    self._seq += 1
    #print(action, self._seq)

    self._sock.send(action + ' ' + str(self._seq) + '\n')

    self.NextLine()

    frame = Frame()
    frame.seq = self.NextInt()
    assert frame.seq == self._seq, 'Expecting %d, got %d' % (self._seq, frame.seq)
    reward = self.NextInt()
    galaxian = self.NextPoint()
    missile = self.NextPoint()
    still_enemies_encoded = []
    for i in xrange(11):
      still_enemies_encoded.append(self.NextInt())
    incoming_enemies = []
    for i in xrange(self.NextInt()):
      incoming_enemies.append(self.NextPoint())
    bullets = []
    for i in xrange(self.NextInt()):
      bullets.append(self.NextPoint())

    return frame, reward

  def NextLine(self):
    self._line = self._fin.readline().strip()
    self._ints = map(int, self._line.split())
    self._idx = 0

  def NextInt(self):
    self._idx += 1
    return self._ints[self._idx - 1]

  def NextPoint(self):
    return Point(self.NextInt(), self.NextInt())


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


# Deep Learning Params
HUMAN_PLAY = 0 # 2160  # 3 minutes
LEARNING_RATE = 0.99
INITIAL_EPSILON = 0.2 # 1.0
FINAL_EPSILON = 0.05
EXPLORE_STEPS = 500000
OBSERVE_STEPS = 1000 # 50000
REPLAY_MEMORY = 50000
MINI_BATCH_SIZE = 100
TRAIN_INTERVAL = 24

CHECKPOINT = 'galaxian2.cpk'
SAVE_INTERVAL = 10000


def Run():
  memory = deque()
  nn = NeuralNetwork()
  game = Game()
  frame, reward = game.Step('stay')

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    save_path = tf.train.latest_checkpoint('.')
    if save_path is not None:
      saver.restore(sess, save_path)
      print("Restored from", save_path)

    steps = 0
    epsilon = INITIAL_EPSILON
    while True:
      if steps < HUMAN_PLAY:
        action = 'human'
      elif random.random() <= epsilon:
        action = OUTPUT_NAMES[random.randrange(OUTPUT_DIM)]
      else:
        q_val = nn.Eval([frame])[0]
        action = OUTPUT_NAMES[np.argmax(q_val)]

      frame1, reward = game.Step(action)

      action_val = np.zeros([OUTPUT_DIM], dtype=np.int)
      action_val[frame1.action_id] = 1

      memory.append((frame, action_val, reward, frame1))
      if len(memory) > REPLAY_MEMORY:
        memory.popleft()

      if steps % TRAIN_INTERVAL == 0 and steps > OBSERVE_STEPS:
        mini_batch = random.sample(memory, min(len(memory), MINI_BATCH_SIZE))
        nn.Train(mini_batch)

      frame = frame1
      steps += 1
      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE_STEPS

      if steps % SAVE_INTERVAL == 0:
        save_path = saver.save(sess, CHECKPOINT, global_step = steps)
        print("Saved to", save_path)

      print("Step %d epsilon: %.6f action: %5s reward: %g" %
          (steps, epsilon, frame.action, reward))


if __name__ == '__main__':
  TestGame()
