from __future__ import division
import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

env = gym.make('Reversi8x8-v0')


class Qnetwork:
    def __init__(self):
        self.path = "./REVERSI_doubledueling_dqn(3)"
        self.env = gym.make('Reversi8x8-v0')
        self.batch_size = 32  # How many experiences to use for each training step.
        self.update_freq = 4  # How often to perform a training step.
        self.y = .99  # Discount factor on the target Q-values
        self.startE = 1  # Starting chance of random action
        self.endE = 0.1  # Final chance of random action
        self.annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
        self.num_episodes = 5000  #10000 How many episodes of game environment to train network with.
        self.pre_train_steps = 10000  #10000 How many steps of random  before training begins.
        self.max_epLength = 64  # The max allowed length of our episode.
        self.load_model = False  # Whether to load a saved model.
        self.h_size = 128  # The size of the final convolutional layer before splitting it into Advantage and Value streams.
        self.tau = 0.001  # Rate to update target network toward primary network
        self.buffer = []
        self.buffer_size = 50000
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.accept = tf.placeholder(shape=[None, 65], dtype=tf.float32)  # None 代表几行
        self.scalarInput = tf.placeholder(shape=[None, 192], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 8, 8, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[3, 3], stride=[1, 1], padding='SAME',biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[2, 2], stride=[1, 1], padding='SAME',biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[2, 2], stride=[1, 1], padding='SAME',biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=self.h_size, kernel_size=[2, 2], stride=[1, 1], padding='SAME',biases_initializer=None)
        self.pool = slim.avg_pool2d(self.conv4, kernel_size=[8, 8], stride=1)
        # This is for dueling network
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.pool, 2, 3)
        # one is for state one is for action 64;64
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()

        self.AW = tf.Variable(xavier_init([self.h_size // 2, 65]), name="AW")  # weight 64*65
        self.VW = tf.Variable(xavier_init([self.h_size // 2, 1]), name='VW')  # value weight 64*1
        # import pdb;pdb.set_trace()
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        self.predict = tf.argmax(tf.abs(self.Qout * self.accept), 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 65, dtype=tf.float32)
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def train(self):
        tf.reset_default_graph()
        mainQN = Qnetwork()  # [Separate Target Network;double DQN]
        targetQN = Qnetwork()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        trainables = tf.trainable_variables()
        targetOps = self.updateTargetGraph(trainables)

        # Set the rate of random action decrease.
        e = self.startE
        stepDrop = (self.startE - self.endE) / self.annealing_steps

        # create lists to contain total rewards and steps per episode
        jList = []
        rList = []
        total_steps = 0

        # Make a path for our model to be saved in.
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        with tf.Session() as sess:
            sess.run(init)
            #if self.load_model == True:
            #    print('Loading Model...')
            #    ckpt = tf.train.get_checkpoint_state(self.path)
            #    saver.restore(sess, ckpt.model_checkpoint_path)
            for kb in range(self.num_episodes):
                #episodeBuffer = self.experience_buffer()
                # Reset environment and get first new observation
                s = env.reset()
                s = self.processState(s)
                d = False
                rAll = 0
                j = 0
                # The Q-Network
                while j < self.max_epLength:  # If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                    j += 1
                    enable = env.possible_actions
                    if 65 in enable:
                        accept = np.zeros(65)
                        accept[64] = 1
                        accept = np.array([accept])
                    else:
                        accept = np.zeros(65)
                        for k in enable:
                            accept[k] = 1
                        accept = np.array([accept])
                    # Choose an action by greedily (with e chance of random action) from the Q-network
                    if np.random.rand(1) < e or total_steps < self.pre_train_steps:
                        enable = env.possible_actions
                        a = np.random.choice(enable)
                    else:
                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s], mainQN.accept: accept})[0]
                        sess.run(mainQN.Qout, feed_dict={mainQN.scalarInput: [s]}) * accept
                    if j % 2 == 1:
                        a = [a, 0]
                    else:
                        a = [a, 1]
                    s1, r, d, _ = env.step(a)
                    s1 = self.processState(s1)
                    total_steps += 1
                    a = a[0]
                    self.add(np.reshape(np.array([s, a, accept, r, s1, d]), [1,6]))  # Save the experience to our episode buffer. [experience replay]
                    if total_steps > self.pre_train_steps:
                        if e > self.endE:
                            e -= stepDrop
                        if total_steps % (self.update_freq) == 0:
                            #import pdb; pdb.set_trace()# Get a random batch of experiences.
                            trainBatch = self.sample(self.batch_size)
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                                     mainQN.accept: np.vstack(trainBatch[:, 2])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 0])})
                            end_multiplier = -(trainBatch[:, 5] - 1)
                            doubleQ = Q2[range(self.batch_size), Q1]
                            targetQ = trainBatch[:, 3] + (self.y * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                                        mainQN.targetQ: targetQ,
                                                                        mainQN.actions: trainBatch[:, 1]})
                            self.updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                    rAll += r
                    s = s1
                    if d == True:
                        if r ==0:
                            print (r)
                        #    #import pdb;pdb.trace()
                        break
                print("{} episode has done, timestep {}".format(kb, j))
                self.add(self.buffer)
                jList.append(j)
                rList.append(rAll)
                # Periodically save the model.
                if len(rList) % 10 == 0:
                    print('下了几局', kb, "一共下了多少次了~~?", total_steps, np.mean(rList[-10:]), e)
            #if kb % 200 == 0:
                # saver.save(sess, self.path + '/model-' + str(kb) + '.ckpt')
            print("Saved Model")
            saver.save(sess, self.path + '/model-' + str(kb) + '.ckpt')
        print("Percent of succesful episodes: " + str(sum(rList) / self.num_episodes) + "%")

    def test_for_white(self, sess):
        b_win=0
        w_win=0
        max_epochs = 100
        for i_episode in range(max_epochs):
            state=env.reset()
            #env.render()
            for t in range(64):
                action = [1, 2]
                enables = env.possible_actions
                if t % 2 == 0:
                    action_ = random.choice(enables)  # 0 表示黑棋
                    action[0] = action_
                    action[1] = 0  # 黑棋 为 0
                else:
                    player=1
                    action_ = self.place(state.reshape([1,192]), player,enables,sess)  # 调用自己训练的模型
                    action[0] = action_
                    if action[0]==64:
                        action[0]=65
                    action[1] = player  # 白棋 为 1
                    #print(enables)
                    #import pdb;pdb.set_trace()
                state, reward, done, _ = env.step(action)

                if done:  # 游戏 结束
                    print("Episode finished after {} timesteps".format(t + 1))
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(env.state[1, :, :] == 1)[0])
                    print('黑',black_score ,'vs','白',white_score)
                    if reward==1:
                        print("黑棋赢了！")
                        b_win +=1
                    elif reward==-1:
                        print("白棋赢了！")
                        w_win+=1
                    break
                    # env.render()
        print("黑棋：", b_win, "  白棋 ", w_win)

    def place(self, state, player, enable, sess):
        if 65 in enable:
            accept = np.zeros(65)
            accept[64] = 1
            accept = np.array([accept])
        else:
            accept = np.zeros(65)
            for i in enable:
                accept[i] = 1
            accept = np.array([accept])
        predict = sess.run(self.predict, feed_dict={self.scalarInput:state, self.accept: accept})[0]
        return(predict)


    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []  # 5
        self.buffer.extend(experience)  # 3


    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 6])

    def processState(self,states):
        return np.reshape(states, [192])

    def updateTargetGraph(self,tfVars):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * self.tau) + ((1 - self.tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder


    def updateTarget(self,op_holder, sess):
        for op in op_holder:
            sess.run(op)


    def save_model(self, sess, I):
        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, self.path + '/model-' + str(I) + '.ckpt')
        return (sess)


    def load_modell(self, sess, I):
        print('Loading Model...')
        # 重新导入模型
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.path)
        #import pdb ;pdb.set_trace()
        saver.restore(sess, ckpt.model_checkpoint_path)