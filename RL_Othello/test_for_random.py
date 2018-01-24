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
#两者选一：否则计算图会拼在一起
#agent = rl()
agenttwo= ql()

#for i in range(200,2000):
#第一大部分：
def train():
#    for i in range(500,800,300):
    i=300
    agent.init_model()
    agent.train_balck_with_white_alpha_beta(i)


#第二部分，循环测试与random打的结果：
def text():
    sess=tf.Session()
    i=300
    I= 'MODEL'+str(i)
    agent.load_model(sess, I)
    agent.test_for_white(sess)
    agent.test_for_black(sess)

def triann():
    #agenttwo.init_model()
    agenttwo.train()

def test():
    #import pdb;pdb.set_trace()
    sess=tf.Session()
    I=str(200)
    #import pdb;pdb.set_trace()
    agenttwo.load_modell(sess, I)
    agenttwo.test_for_white(sess)


#train()
#text()
#triann()
test()

