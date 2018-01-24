import gym
import random
import numpy as np
import tensorflow as tf

import alpha_beta as ab
from RL_DOUBLE_QG_AGENT import RL_QG_agent as rl
from DOUBLE_DUELING_Q_AGENT import Qnetwork as ql
import os


env = gym.make('Reversi8x8-v0')
env.reset()
max_epochs = 1
w_win =[]
b_win =[]
agent = rl()


#for i in range(200,2000):
#第一大部分：
def train():
#    for i in range(500,800,300):
    i=299
    agent.init_model()
    agent.train_balck_with_white_alpha_beta(i)


#第二部分，循环测试与random打的结果：
def text():
    sess=tf.Session()
    i=299
    I= 'MODEL'+str(i)
    agent.load_model(sess,I)
    agent.test_for_white(sess)
    agent.test_for_black(sess)


#第三部分，测试出来的模型和不同层数的alpha_beta模型跑：
def third():
    b_win=[]
    w_win=[]
    sess=tf.Session()
    for i in [299]:
        I= 'MODEL'+str(i)
        agent.load_model(sess,I)
        black=[]
        white=[]
        for layer in [1,2,3,4,5,6]:
            observation = env.reset()
            action = [1, 2]
            t=0
            while t <= 64:
                if t%2==0:
                    enables = env.possible_actions
                    action_ = ab.place(observation, enables, 0,layer)#  0 表示黑棋
                    action[0] = action_
                    action[1] = 0   # 黑棋 为 0
                else:
                    player = 1
                    enables = env.possible_actions
                    action_ = agent.place(observation, player, enables, sess)  # 调用自己训练的模型
                    if action_==64:
                        action_=65
                    action[0] = action_
                    action[1] = 1  # 白棋 为 1
                observation, reward, done, info = env.step(action)
                t=t+1
                if done:  # 游戏 结束
                    env.render()
                    print("Episode finished after {} timesteps".format(t + 1))
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(env.state[1, :, :] == 1)[0])
                    if black_score > white_score:
                        print("黑棋赢了！")
                        black.append(layer)
                    elif black_score <= white_score:
                        #print("白棋赢了！")
                        white.append(layer)
                    print (white_score)
                    if reward==0:
                        print(reward)
                        import pdb;pdb.set_trace()
                    break
        b_win.append([I,[black]])
        w_win.append([I,[white]])
        print('使用模型',I,'和ab训练得到的结果的报告呀~~~~~~'+"a-btree 赢的层数：",b_win,"  自己的agent_q赢的层数",w_win)
    print('For white_agent',b_win,w_win)


# 第三部分，测试出来的模型和不同层数的alpha_beta模型跑：
def forth():
    b_win = []
    w_win = []
    sess = tf.Session()
    for i in [299]:
        I = 'MODEL' + str(i)
        agent.load_model(sess, I)
        black = []
        white = []
        for layer in [1, 2, 3, 4, 5, 6]:
            observation = env.reset()
            action = [1, 2]
            t = 0
            while t <= 64:
                if t % 2 == 1:
                    enables = env.possible_actions
                    action_ = ab.place(observation, enables, 1, layer)  # 0 表示黑棋
                    action[0] = action_
                    action[1] = 1  # 棋 为 0
                else:
                    player = 0
                    enables = env.possible_actions
                    action_ = agent.place(observation, player, enables, sess)  # 调用自己训练的模型
                    if action_==64:
                        action_=65
                    action[0] = action_
                    action[1] = 0  # 白棋 为 1
                observation, reward, done, info = env.step(action)
                t = t + 1
                if done:  # 游戏 结束
                    env.render()
                    print("Episode finished after {} timesteps".format(t + 1))
                    black_score = len(np.where(env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(env.state[1, :, :] == 1)[0])
                    if black_score > white_score:
                        print("黑棋赢了！")
                        black.append(layer)
                    elif black_score <= white_score:
                        #print("白棋赢了！")
                        white.append(layer)
                    if reward==0:
                        print(reward)
                        import pdb;pdb.set_trace()
                    print(black_score)
                    break
        b_win.append([I, [black]])
        w_win.append([I, [white]])
        print('使用模型', I, '和ab训练得到的结果的报告呀~~~~~~' + "  自己的agent_q赢的层数", b_win,"a-btree 赢的层数：" , w_win)
    print('For Black_agent', b_win)

#train()
#text()
third()
forth()