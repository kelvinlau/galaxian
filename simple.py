from __future__ import print_function
from collections import deque
import os
import sys
import random
import time
import pygame
from pygame.locals import *
import numpy as np
import tensorflow as tf


PLAY = False

SIDE = 8
INPUT_DIM = (1+SIDE) * SIDE
SIZE = 50

ACTION_NAMES = ['_', 'L', 'R']
ACTION_ID = {'_': 0, 'L': 1, 'R': 2}
OUTPUT_DIM = len(ACTION_NAMES)

# Hyperparameters.
GAMMA = 0.99
INITIAL_EPSILON = 1
FINAL_EPSILON = 0.1
EXPLORE_STEPS = 100000
OBSERVE_STEPS = 0
REPLAY_MEMORY = 100000
MINI_BATCH_SIZE = 32
TRAIN_INTERVAL = 1
UPDATE_TARGET_NETWORK_INTERVAL = 1000

DOUBLE_Q = True
if DOUBLE_Q:
  FINAL_EPSILON = 0.01

if PLAY:
  INITIAL_EPSILON = 0

# Checkpoint.
CHECKPOINT_DIR = 'simple3/'
CHECKPOINT_FILE = 'model.ckpt'
SAVE_INTERVAL = 1000


class Frame:
  def __init__(self, action, reward, terminal, x, data):
    self.action = action
    self.action_id = ACTION_ID[self.action]
    self.reward = reward
    self.terminal = terminal
    row = np.zeros((1, SIDE))
    row[0][x] = 1
    self.data = np.reshape(np.append(data, row, axis=0), INPUT_DIM)


class Game:
  def __init__(self):
    self.x = random.randint(0, SIDE-1)
    self.data = np.zeros((SIDE, SIDE))
    self.score = 0

    pygame.init()
    self.clock = pygame.time.Clock()
    self.screen = pygame.display.set_mode((SIDE*SIZE, (SIDE+1)*SIZE))
    pygame.display.set_caption('Simple Game')

  def Step(self, action):
    # Comsume pygame events.
    for event in pygame.event.get():
      if event.type == QUIT:
        pygame.quit()
        sys.exit()

    if PLAY:
      time.sleep(0.2)

    reward = self.data[-1][self.x]
    row = np.array([random.randint(-1, 1) for i in xrange(SIDE)])
    self.data = np.append([row], self.data[:-1], axis=0)
    if action == 'L' and self.x - 1 >= 0:
      self.x -= 1
    if action == 'R' and self.x + 1 < SIDE:
      self.x += 1
    terminal = False
    self.score += reward

    BLACK = (0, 0, 0)
    WHITE = (127, 127, 127)
    GREEN = (0, 127, 0)
    RED = (127, 0, 0)
    self.screen.fill(BLACK)
    pygame.draw.rect(self.screen, WHITE, pygame.Rect(self.x*SIZE, SIDE*SIZE, SIZE, SIZE))
    for y in xrange(SIDE):
      for x in xrange(SIDE):
        if self.data[y][x] != 0:
          pygame.draw.rect(self.screen, GREEN if self.data[y][x] > 0 else RED,
              pygame.Rect(x*SIZE, y*SIZE, SIZE, SIZE))
    score = pygame.font.Font(None, 15).render("%d" % self.score, 1, (255,255,0))
    self.screen.blit(score, (10, 10))
    pygame.display.update()
    self.clock.tick(60)

    return Frame(action, reward, terminal, self.x, self.data)


def TestGame():
  g = Game()
  while True:
    action = raw_input().strip()
    g.Step(action)


def ClippedError(x):
  # Huber loss
  return tf.select(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)


class NeuralNetwork:
  def __init__(self, name, trainable=True):
    var = lambda shape: tf.Variable(
        tf.truncated_normal(shape, stddev=.02), trainable=trainable)

    with tf.variable_scope(name):
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

    self.theta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    assert len(self.theta) == 6, len(self.theta)

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
        self.input: [f.data for f in frames]
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
        self.input: [f.data for f in frame_batch],
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
  frame = game.Step('_')

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

      print("Step %d epsilon: %.6f nn: %s q: %-33s action: %s reward: %2.0f "
          "cost: %8.3f y: %8.3f" %
          (steps, epsilon, FormatList(nn.CheckSum()), FormatList(q_val),
            frame1.action, frame1.reward, cost, y_val))


if __name__ == '__main__':
  Run()
