# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
""" 

from __future__ import print_function
import numpy as np
import time
import sys
import bitboard

class Board(object):
    """
    board for the game
    """

    def __init__(self, env, **kwargs):
        self.env = env
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.players = [0, 1] # player1 and player2
        
    def init_board(self, start_player=0):
        observation = self.env.reset()
        self.black, self.white = bitboard.board_to_bit(observation)
        self.done = False
        self.curr_reward = 0
        self.current_player = self.players[start_player]  # start player        
        #self.availables = list(range(self.width * self.height)) # available moves
        #self.availables = self.env.possible_actions
        self.states = {} # board states, key:move as location on the board, value:player as pieces type
        self.last_move = -1
        self.availables = bitboard.find_correct_moves(self.black, self.white)

    def current_state(self): 
        """return the board state from the perspective of the current player
        shape: 4*width*height"""
        raw_state = bitboard.bit_to_board(self.black, self.white) 
        square_state = np.zeros((3, self.width, self.height))
        if self.current_player == 0:
            square_state[0:2] = raw_state[0:2]
        else:
            square_state[0:2] = raw_state[1::-1]
        if self.last_move < self.width * self.height:
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0  # last move indication, if 65 then no indication
        return square_state

    def do_move(self, move):
        self.states[move] = self.current_player
        if self.current_player == self.players[0]:
            self.black, self.white = bitboard.step(move, self.black, self.white)
            self.availables = bitboard.find_correct_moves(self.white, self.black)
        else:
            self.white, self.black = bitboard.step(move, self.white, self.black)
            self.availables = bitboard.find_correct_moves(self.black, self.white)
        self.done = bitboard.is_finished(self.black, self.white)
        self.current_player = self.players[0] if self.current_player == self.players[1] else self.players[1] 
        self.last_move = move
        if self.done:
            black_cnt = bitboard.bit_count(self.black)
            white_cnt = bitboard.bit_count(self.white)
            self.curr_reward = 1.0 if black_cnt > white_cnt else -1.0

    def has_a_winner(self):
        if self.done == False:
            return False, -1
        player = 0 if self.curr_reward == 1.0 else 1
        return True, player

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """
    game server
    """

    def __init__(self, board, **kwargs):
        self.board = board

    def start_play(self, player1, player2, start_player=0):
        """
        start a game between two players
        """
        if start_player not in (0,1):
            raise Exception('start_player should be 0 (player1 first) or 1 (player2 first)')
        self.board.init_board()
        p1, p2 = self.board.players
        if p1 == start_player:
            player1.set_player_ind(p1)
            player2.set_player_ind(p2)
        else:
            player1.set_player_ind(p2)
            player2.set_player_ind(p1)
        players = {p1: player1, p2:player2} if p1 == start_player else {p1: player2, p2:player1}
        while(1):
            #import pdb;pdb.set_trace()
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            start = time.time()
            move = player_in_turn.get_action(self.board)
            print("Player %d move %d elapsed time %f s" % (current_player, move, time.time()-start))
            self.board.do_move(move)
            end, winner = self.board.game_end()
            if end:
                return 1 if winner == start_player else -1
            
            
    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree
        store the self-play data: (state, mcts_probs, z)
        """
        self.board.init_board()
        states, mcts_probs, current_players = [], [], []        
        cnt = 0
        while(1):
            if cnt < 30:
                temp = 1
            else:
                temp = 1e-3
            #print("Current Player: %d" % self.board.get_current_player())
            #print("Possible actions: ", self.board.availables)
            move, move_probs = player.get_action(self.board, temp=temp, return_prob=1)
            # store the data
            #print("Move chosen: %d" % (move))
            states.append(self.board.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            # perform a move
            self.board.do_move(move)
            end, winner = self.board.game_end()
            cnt += 1
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))  
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                #reset MCTS root node
                player.reset_player()
                return winner, list(zip(states, mcts_probs, winners_z))
            
