import os
#from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
#from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import torch
import gym
from game import Board, Game
import numpy as np
from bitboard import bit_to_board

class RL_QG_agent(object):
    def __init__(self):
        self.temp = 1e-3 # the temperature param
        self.n_playout = 200 # num of simulations for each move
        self.c_puct = 5
        self.board_width = 8
        self.board_height = 8
        self.model_path = os.path.join("./models/curr_model_100rollout.pt")
        #self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, net_params=None)
        #self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn, c_puct=self.c_puct, n_playout=self.n_playout)
        self.mcts_player = MCTS_Pure(c_puct=5, n_playout=self.n_playout)
        self.env = gym.make("Reversi8x8-v0")
        self.init_model()
        #self.load_model()

    def init_model(self):
        self.board = Board(env=self.env, width=self.board_width, height=self.board_height)
        self.board.init_board()
        self.game = Game(self.board)
        self.have_step = False

    def place(self, state, enables, player=None):
        curr_state = bit_to_board(self.board.black, self.board.white)
        curr_state = 1 - (curr_state[0] + curr_state[1])
        reverse_change = np.where((curr_state - state[2]) == -1)
        if self.have_step == False:
            pass
        elif reverse_change[0].shape[0] > 1:
            self.board.init_board()
            self.have_step = False
        curr_state = bit_to_board(self.board.black, self.board.white)
        curr_state = 1 - (curr_state[0] + curr_state[1])
        change = np.where((curr_state - state[2]) == 1)
        if change[0].shape[0] == 1:
            action = change[0][0] * self.board_width + change[1][0]
            self.board.do_move(action)
        else:
            if self.have_step == False:
                pass
            else:
                action = 65
                self.board.do_move(action)

        move = self.mcts_player.get_action(self.board)
        self.board.do_move(move)
        self.have_step = True

        return move

    def load_model(self):
        self.policy_value_net.policy_value_net.load_state_dict(torch.load(self.model_path))

