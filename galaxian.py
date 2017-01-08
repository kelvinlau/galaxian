"""Galaxian NN trainer.

Read data on human playing Galaxian, train a Neural Network.
"""

from __future__ import print_function
from collections import defaultdict
import random
import time
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
    self.gx = 0.0
    self.m = 0.0
    self.output_id = None


# Deep Learning Params
LEARNING_RATE = 0.99
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
EXPLORE = 500000
OBSERVE = 50000
REPLAY_MEMORY = 500000
BATCH = 100


class NeuralNetwork:
  def __init__():
    # Still enemies.
    self.input_s = tf.placeholder("float",
        [None, WIDTH, NUM_SNAPSHOTS])

    # Incoming enemies.
    self.input_i = tf.placeholder("float",
        [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])

    # Incoming bullets.
    self.input_b = tf.placeholder("float",
        [None, WIDTH, HEIGHT, NUM_SNAPSHOTS])

    # 0-d values: galaxian_x, missile_y.
    self.input_c = tf.placeholder("float", [None, 2])

    # 96 x 32 x 6
    input_ib = tf.concat(3, [self.input_i, self.input_b])

    zeros_var = lambda shape: tf.Variable(tf.zeros(shape))

    # 23 x 15 x 32
    conv1 = tf.nn.relu(tf.nn.conv2d(
      input_ib, zero_var([8, 4, 6, 32]), strides = [1, 4, 2, 1],
      padding = "VALID")
      + zero_var([32]))

    # 10 x 7 x 64
    conv2 = tf.nn.relu(tf.nn.conv2d(
      conv1, zero_var([4, 4, 32, 64]), strides = [1, 2, 2, 1],
      padding = "VALID")
      + zero_Var([64]))

    # 9 x 6 x 64
    conv3 = tf.nn.relu(tf.nn.conv2d(
      conv2, zero_var([2, 2, 64, 64]), strides = [1, 1, 1, 1],
      padding = "VALID")
      + zero_Var([64]))

    # 3456
    conv3_flat = tf.reshape(conv3, [-1, 3456])

    # 288
    # TODO: Conv input_s too?
    input_s_flat = tf.reshape(self.input_s, [-1, WIDTH*NUM_SNAPSHOTS])

    # 3746
    all_flat = tf.concat(1, [conv3_flat, input_s_flat, self.input_c])

    fc4 = tf.nn.relu(
        tf.matmul(conv3_flat, zero_var([3746, 784])) + zero_var([784]))

    self.output_layer = (tf.matmul(fc4, zero_var([784, OUTPUT_DIM])) +
        zero_var([OUTPUT_DIM]))

    self.argmax = tf.placeholder("float", [None, output])
    self.gt = tf.placeholder("float", [None])
    self.action = tf.reduce_sum(tf.mul(predict_action, argmax), reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(action - gt))
    self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(cost)


class Game:
  def __init__(self):
    self._fout = open("galaxian.ctrl", "w")
    self._fin = None  # Lazily open.
    self._seq = 0

  def Step(self, ctrl):
    self._seq += 1
    print(ctrl, self._seq)
    self._fout.write(ctrl + ' ' + str(self._seq) + '\n')
    self._fout.flush()

    while self._fin is None:
      time.sleep(0.1)
      self._fin = open("galaxian.frames", "r")

    frame = Frame()
    frame.seq = int(self.NextLine())
    assert frame.seq == self._seq

    frame.ctrl = self.NextLine()
    assert frame.ctrl == ctrl
    frame.ctrl_id = OUTPUT_IDS[frame.ctrl]

    while True:
      key = self.NextLine()
      if key == 'done':
        break
      val = float(self.NextLine())
      if key == 'gx':
        frame.gx = val
      elif key == 'm':
        frame.m = val
      elif key[0] == 's':
        gid = int(key[1])
        ix = int(key[3:]) + WIDTH/2
        frame.s[ix][gid] = val
      else:
        assert key[0] == 'i' or key[0] == 'b'
        gid = int(key[1])
        ix, iy = map(key[3:].split(','), int)
        ix += WIDTH/2
        a = frame.i if key[0] == 'i' else frame.b
        a[ix][iy][gid] = val

    return frame

  def NextLine(self):
    while True:
      line = self._fin.readline().strip()
      if line: break
      time.sleep(0.1)
    return line.strip()


def ShowFrames(frames):
  print("Read %d frames" % len(frames))
  print("Random 10 frames:")
  for frame in random.sample(frames, 10):
    print("Output:", frame.output_id)
  output_summary = defaultdict(int)
  for _, outputs in frames:
    output_summary[OUTPUT_NAME_MAP[frame.output_id]] += 1
  print("Output summary:")
  for buttons, freq in output_summary.items():
    print(1.0*freq/len(frames), freq, buttons)


def run():
  nn = NeuralNetwork()

  game = Game()
  D = deque()

  with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    saver = tf.train.Saver()

    n = 0
    epsilon = INITIAL_EPSILON
    while True:
      action_t = predict_action.eval(feed_dict = {input_image : [input_image_data]})[0]

      argmax_t = np.zeros([output], dtype=np.int)
      if(random.random() <= INITIAL_EPSILON):
        maxIndex = random.randrange(output)
      else:
        maxIndex = np.argmax(action_t)
      argmax_t[maxIndex] = 1
      if epsilon > FINAL_EPSILON:
        epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

      reward, image = game.step(list(argmax_t))

      image = cv2.cvtColor(cv2.resize(image, (100, 80)), cv2.COLOR_BGR2GRAY)
      ret, image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
      image = np.reshape(image, (80, 100, 1))
      input_image_data1 = np.append(image, input_image_data[:, :, 0:3], axis = 2)

      D.append((input_image_data, argmax_t, reward, input_image_data1))

      if len(D) > REPLAY_MEMORY:
        D.popleft()

      if n > OBSERVE:
        minibatch = random.sample(D, BATCH)
        input_image_data_batch = [d[0] for d in minibatch]
        argmax_batch = [d[1] for d in minibatch]
        reward_batch = [d[2] for d in minibatch]
        input_image_data1_batch = [d[3] for d in minibatch]

        gt_batch = []
        out_batch = predict_action.eval(feed_dict = {input_image : input_image_data1_batch})
        for i in range(0, len(minibatch)):
          gt_batch.append(reward_batch[i] + LEARNING_RATE * np.max(out_batch[i]))

        optimizer.run(feed_dict = {gt : gt_batch, argmax : argmax_batch, input_image : input_image_data_batch})

      input_image_data = input_image_data1
      n = n+1

      if n % 10000 == 0:
        saver.save(sess, 'game.cpk', global_step = n)

      print(n, "epsilon:", epsilon, " " ,"action:", maxIndex, " " ,"reward:", reward)


if __name__ == '__main__':
  game = Game()
  while True:
    for i in xrange(10):
      game.Step('left')
    game.Step('fire')
    for i in xrange(10):
      game.Step('stay')
    for i in xrange(10):
      game.Step('right')
    game.Step('fire')
    for i in xrange(10):
      game.Step('stay')
