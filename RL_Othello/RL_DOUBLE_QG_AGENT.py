import tensorflow as tf
import os
import gym
import random
import numpy as np
import alpha_beta as ab
import matplotlib.pyplot as plt
#黑棋子是0，白棋子是1
#返回的是一个action
#第一张棋盘是黑棋子0
#第二张棋盘是白棋子1

class RL_QG_agent:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REVERSI_double_sqn(2)")
        self.env = gym.make('Reversi8x8-v0')
        #self.sess = tf.Session()
        #tf.reset_default_graph()
        self.init_model()


    def init_model(self):
        self.accept=tf.placeholder(shape=[1,65], dtype=tf.float32)
        self.inputs1 = tf.placeholder(shape=[1, 64], dtype=tf.float32)
        self.W_black = tf.Variable(tf.random_uniform([64, 65], 0.5, 0.01))
        self.W_white = tf.Variable(tf.random_uniform([64, 65], -0.5, 0.01))

        self.Qout_black = tf.matmul(self.inputs1, self.W_black)  # [1,65]
        self.Qout_white = tf.matmul(self.inputs1, self.W_white)

        self.predict_black=tf.argmax(tf.abs(self.Qout_black*self.accept), axis=1)
        self.predict_white=tf.argmax(tf.abs(self.Qout_white*self.accept), axis=1)

        self.nextQ_black = tf.placeholder(shape=[1, 65], dtype=tf.float32)
        self.nextQ_white = tf.placeholder(shape=[1, 65], dtype=tf.float32)


        self.loss_black= tf.reduce_sum(tf.square(self.nextQ_black - self.Qout_black))
        self.loss_white= tf.reduce_sum(tf.square(self.nextQ_white - self.Qout_white))

        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

        self.updateModel_black = self.trainer.minimize(self.loss_black)
        self.updateModel_white = self.trainer.minimize(self.loss_white)

        self.e = 0.1
        self.init = tf.initialize_all_variables()


    def train(self,num_episodes):
        rList_BLACK = []
        rList_WHITE = []
        y = 0.99
        I='MODEL'+str(num_episodes)
        with tf.Session() as sess:
            sess.run(self.init)
            for ZZZ in range(num_episodes):
                s = self.env.reset()
                rAll_BLACK=0
                rAll_WHITE=0
                j = 0
                while j < 64:
                    input1 = self.transfer(s)
                    # import pdb;pdb.set_trace()
                    j += 1
                    enable = self.env.possible_actions
                    if 65 in enable:
                        accept = np.zeros(65)
                        accept[64] = 1
                        accept = np.array(accept)
                    else:
                        accept = np.zeros(65)
                        for k in enable:
                            accept[k] = 1
                    if j %2==1:
                        allQ_black = sess.run(self.Qout_black, feed_dict={self.inputs1: input1})
                        predict_black = sess.run(self.predict_black,feed_dict={self.inputs1: input1,self.accept:accept.reshape([1, -1])})[0]
                        #import pdb;pdb.set_trace()
                        action_black= np.zeros(2).astype(int)
                        action_black[0] = predict_black
                        if random.randint(1, 10) == 1:
                            action_black[0] = random.choice(self.env.possible_actions)  # 0.1 十分之一的概率变动作。
                        action_black[1] = 0
                        s1_black, r_black, done, _ = self.env.step(action_black)
                    else:
                        allQ_white = sess.run(self.Qout_white, feed_dict={self.inputs1: input1})
                        predict_white = sess.run(self.predict_white, feed_dict={self.inputs1: input1, self.accept: accept.reshape([1, -1])})[0]
                        action_white = np.zeros(2).astype(int)
                        action_white[0] = predict_white
                        if random.randint(1, 10) == 1:
                            action_white[0] = random.choice(self.env.possible_actions)  # 0.1 十分之一的概率变动作。
                        action_white[1] = 1
                        s1_white, r_white, done, _ = self.env.step(action_white)
                    if j%2 ==0: #2次一UPDATE
                        input_black = self.transfer(s1_black)
                        input_white= self.transfer(s1_white)

                        Q1_black= sess.run(self.Qout_black, feed_dict={self.inputs1: input_white})
                        #print ("q1_black",Q1_black)
                        Q1_white = sess.run(self.Qout_white, feed_dict={self.inputs1: input_black})
                        #print ("q1_white",Q1_white)
                        maxq1_black=np.max(Q1_black)
                        minq1_white=np.min(Q1_white)
                        targetQ_black = allQ_black
                        targetQ_white = allQ_white

                        action_black[0] = 64 if action_black[0] == 65 else action_black[0]
                        action_white[0] = 64 if action_white[0] == 65 else action_white[0]
                        targetQ_black[0, action_black[0]] = r_black + y * maxq1_black
                        targetQ_white[0, action_white[0]] = r_black + y * minq1_white

                        _, W_white, loss_white = sess.run([self.updateModel_white, self.W_white, self.loss_white],feed_dict={self.inputs1: input1, self.nextQ_white: targetQ_black})
                        _, W_black, loss_black = sess.run([self.updateModel_black, self.W_black, self.loss_black],feed_dict={self.inputs1: input1, self.nextQ_black: targetQ_white})

                        rAll_BLACK += r_black
                        rAll_WHITE += r_white
                        s = s1_white

                    if done == True:
                        print(loss_black, loss_white)
                        if r_black == 1 or r_white==1:
                            print("黑棋赢了")
                        elif r_white==-1 or r_black==-1:
                            print("白棋赢了")
                        else:
                            #print (r_white,r_black)
                            if len(np.where(self.env.state[0, :, :] == 1)[0])>len(np.where(self.env.state[1, :, :] == 1)[0]):
                                print("黑棋赢了！")
                            elif len(np.where(self.env.state[0, :, :] == 1)[0])<len(np.where(self.env.state[1, :, :] == 1)[0]):
                                print("白棋赢了！")
                        self.e = 1. / ((ZZZ / 50) + 10)  # 至少0.1的概率                            print('第%d局训练好了~好开心呀~' % (ZZZ)
                        print("{} Episode finished after {} timesteps".format(ZZZ,j))
                        break
                rList_BLACK.append(rAll_BLACK)
                rList_WHITE.append(rAll_WHITE)
            self.save_model(sess,I)
        #plt.plot(rList_BLACK)
        #plt.show()

    def train_balck_with_white_alpha_beta(self, num_episodes):
        rList_BLACK = []
        rList_WHITE = []
        y = 0.99
        I = 'MODEL' + str(num_episodes)
        with tf.Session() as sess:
            sess.run(self.init)
            for ZZZ in range(num_episodes):
                s = self.env.reset()
                rAll_BLACK = 0
                rAll_WHITE = 0
                j = 0
                while j < 64:
                    input1 = self.transfer(s)
                    # import pdb;pdb.set_trace()
                    j += 1
                    enable = self.env.possible_actions
                    if 65 in enable:
                        accept = np.zeros(65)
                        accept[64] = 1
                        accept = np.array(accept)
                    else:
                        accept = np.zeros(65)
                        for k in enable:
                            accept[k] = 1
                    if j % 2 == 1:
                        allQ_black = sess.run(self.Qout_black, feed_dict={self.inputs1: input1})
                        predict_black = sess.run(self.predict_black, feed_dict={self.inputs1: input1,self.accept: accept.reshape([1, -1])})[0]
                        # import pdb;pdb.set_trace()
                        action_black = np.zeros(2).astype(int)
                        action_black[0] = predict_black
                        if random.randint(1, 10) == 1:
                            action_black[0] = random.choice(self.env.possible_actions)  # 0.1 十分之一的概率变动作。
                        action_black[1] = 0
                        s1_black, r_white, done, _ = self.env.step(action_black)
                    else:
                        allQ_white = sess.run(self.Qout_white, feed_dict={self.inputs1: input1})
                        action_white = np.zeros(2).astype(int)
                        action_white[0] = ab.place(s,enable,1,3)
                        action_white[1] = 1
                        s1_white, r_black, done, _ = self.env.step(action_white)
                    if j % 2 == 0:  # 2次一UPDATE
                        input_black = self.transfer(s1_black)
                        input_white = self.transfer(s1_white)
                        Q1_black = sess.run(self.Qout_black, feed_dict={self.inputs1: input_white})
                        # print ("q1_black",Q1_black)
                        Q1_white = sess.run(self.Qout_white, feed_dict={self.inputs1: input_black})
                        # print ("q1_white",Q1_white)
                        maxq1_black = np.max(Q1_black)
                        minq1_white = np.min(Q1_white)
                        targetQ_black = allQ_black
                        targetQ_white = allQ_white

                        action_black[0] = 64 if action_black[0] == 65 else action_black[0]
                        action_white[0] = 64 if action_white[0] == 65 else action_white[0]
                        targetQ_black[0, action_black[0]] = r_black + y * maxq1_black
                        targetQ_white[0, action_white[0]] = r_white + y * minq1_white

                        _, W_white, loss_white = sess.run([self.updateModel_white, self.W_white, self.loss_white],
                                                          feed_dict={self.inputs1: input1,
                                                                     self.nextQ_white: targetQ_black})
                        _, W_black, loss_black = sess.run([self.updateModel_black, self.W_black, self.loss_black],
                                                          feed_dict={self.inputs1: input1,
                                                                     self.nextQ_black: targetQ_white})

                        s = s1_white

                    if done == True:
                        #print(loss_black, loss_white)
                        if r_black == 1 or r_white == 1:
                            rAll_BLACK += r_black+r_white
                            print("黑棋赢了")
                            print('r_black',r_black,'r_white',r_white)
                        elif r_white == -1 or r_black== -1:
                            rAll_BLACK += r_black+r_white
                            print("白棋赢了")
                            print('r_black',r_black,'r_white',r_white)
                        else:
                            if len(np.where(self.env.state[0, :, :] == 1)[0])>len(np.where(self.env.state[1, :, :] == 1)[0]):
                                print("黑棋赢了！")
                                rAll_BLACK+=1
                            elif len(np.where(self.env.state[0, :, :] == 1)[0])<len(np.where(self.env.state[1, :, :] == 1)[0]):
                                print("白棋赢了！")
                                rAll_BLACK+=-1
                        self.e = 1. / ((ZZZ / 50) + 10)  # 至少0.1的概率；print('第%d局训练好了~好开心呀~' % (ZZZ)
                        print("{} Episode finished after {} timesteps".format(ZZZ, j))
                        break
                rList_BLACK.append(rAll_BLACK)
            self.save_model(sess, I)
            print (rList_BLACK)
            #看black是否收敛。
            plt.plot(rList_BLACK)
            plt.show()

    def test_for_white(self,sess):
        self.env.reset()
        count = 0
        max_epochs = 100
        for i_episode in range(max_epochs):
            stateobserved = self.env.reset()
            # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
            done = False
            for t in range(64):
                #self.env.render()  # 打印当前棋局
                if t %2 ==0:
                    enables = self.env.possible_actions
                    player=0
                    action_ = random.choice(enables)
                else:
                    enables = self.env.possible_actions
                    player = 1
                    action_ = self.place(observation, player, enables, sess)
                    if action_==64:
                        action_=65
                observation, reward, done, info = self.env.step([action_, player])
                if done:  # 游戏 结束
                    black_score = len(np.where(self.env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(self.env.state[1, :, :] == 1)[0])
                    print('黑',black_score ,'vs','白',white_score)
                    #print("Episode finished after {} timesteps".format(t + 1))
                    if reward==1:
                        pass
                    elif reward==-1:
                        count = count + 1
                    break
            print ("下完了第几局",i_episode+1)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~agent_white赢了多少局呢~~~~~~~~~~~~~~~~~~~~~~~~~~~~",count)

    def test_for_black(self,sess):
        self.env.reset()
        count = 0
        max_epochs = 100
        for i_episode in range(max_epochs):
            observation=self.env.reset()
            for t in range(64):
                #self.env.render()  # 打印当前棋局
                if t %2 ==0:
                    enables = self.env.possible_actions
                    player = 0
                    action_ = self.place(observation, player, enables, sess)
                    if action_==64:
                        action_=65
                else:
                    enables = self.env.possible_actions
                    player = 1
                    action_ = random.choice(enables)
                observation, reward, done, info = self.env.step([action_, player])
                if done:  # 游戏 结束
                    black_score = len(np.where(self.env.state[0, :, :] == 1)[0])
                    white_score = len(np.where(self.env.state[1, :, :] == 1)[0])
                    print('黑', black_score, 'vs', '白', white_score)
                    #print(reward)
                    #print("Episode finished after {} timesteps".format(t + 1))
                    if reward==1:
                        #print("黑棋赢了！")
                        count = count + 1
                    elif reward==-1:
                        pass
                        #print("白棋赢了！")
                    break
            print ("下完了第几局",i_episode+1)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~agent_black赢了多少局呢~~~~~~~~~~~~~~~~~~~~~~~~~~~~",count)

    def place(self,state,player,enable,sess):
            # 这个函数 主要用于测试， 返回的 action是 0-63 之间的一个数值，
            # action 表示的是 要下的位置。
            #sess=tf.Session()
            #self.load_model(sess)
            if 65 in enable:
                accept = np.zeros(65)
                accept[64] = 1
                accept = np.array(accept)
            else:
                accept = np.zeros(65)
                for k in enable:
                    accept[k] = 1
            input1 = self.transfer(state)
            if player==1:
                predict = sess.run(self.predict_white, feed_dict={self.inputs1: input1,self.accept:accept.reshape([1,-1])})[0]
            else:
                predict = sess.run(self.predict_black, feed_dict={self.inputs1: input1,self.accept:accept.reshape([1,-1])})[0]
            predict = 65 if predict == 64 else predict
            if 65 in enable:
                return 65
            return (predict)



    def save_model(self,sess,I):
        # 保存模型
        self.saver = tf.train.Saver()
        self.saver.save(sess, os.path.join(self.model_dir, 'parameter'+str(I)+'.ckpt'))
        return (sess)


    def load_model(self,sess,I):
        print('Loading Model...')
        # 重新导入模型
        self.saver = tf.train.Saver()
        self.saver.restore(sess, os.path.join(self.model_dir, 'parameter'+str(I)+'.ckpt'))

    def transfer(self,state_qipan):  # state_qipan其实是observation
        state = np.zeros((8, 8))
        state = state.reshape(1, 64)
        for i in range(8):
            for j in range(8):
                if (state_qipan[2][i][j] == 0):
                    if (state_qipan[1][i][j] == 1):
                        state[0, (i * 8 + j)] = -1  # 白棋
                        # print(i*8+j)
                    elif (state_qipan[0][i][j] == 1):
                        state[0, (i * 8 + j)] = 1  # 黑棋

        return (state)

    def whether_no_place(self,player):
        print('一方没子下，pass了')
        s, r, done, _ = self.env.step([65, player])
        input1 = self.transfer(s)
        allQ = self.sess.run(self.Qout, feed_dict={self.inputs1: input1})
        print('可以下的棋局这种情况好特殊', self.env.possible_actions)
        enable = self.env.possible_actions
        if 65 in enable:
            s, r, done, _ = self.env.step([65, 0])
            print('双方都没子下了')
            print('这样就可以数数子结束了')
            end=sum(sum(input1))
            print(end)
            if int(end) > 0:
                print('黑棋赢了')
                r=1
                print('end')
                return(s,r,done)
            else:
                print('白棋赢了')
                r=-1
                print('end')
                return (s,r, done)
        else:
            return(s,r,done)
