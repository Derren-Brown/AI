#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/3/2 21:06
# @Author  : yaoqi
# @Email   : yaoqi_isee@zju.edu.cn
# @File    : tic_tac_toe.py

# +++++++++++++++++++++++++++++++++++++++++++++ README ++++++++++++++++++++++++++++++++++++++++
# You will write a tic tac toc player and play game with this computer player.
# You will use MiniMaxSearch with depth limited strategy. For simplicity, Alpha-Beta Pruning
# is not considered.
# Let's make some assumptions.
# 1. The computer player(1) use circle(1) and you(-1) use cross(-1).
# 2. The computer player is MAX user and you are MIN user.
# 3. The miniMaxSearch depth is 3, so that the computer predict one step further. It first
# predicts what you will do if it makes a move and choose a move that maximize its gain.
# 4. You play first
# +++++++++++++++++++++++++++++++++++++++++++++ README ++++++++++++++++++++++++++++++++++++++++

import numpy as np


def MinimaxSearch(current_state):
    """
    Search the next step by Minimax Search with depth limited strategy
    The search depth is limited to 3, computer player(1) uses circle(1) and you(-1) use cross(-1)
    :param current_state: current state of the game, it's a 3x3 array representing the chess board, array element lies
    in {1, -1, 0}, standing for circle, cross, empty place.
    :return: row and column index that computer player will draw a circle on. Note 0<=row<=2, 0<=col<=2
    """
    # -------------------------------- Your code starts here ----------------------------- #
    assert isinstance(current_state, np.ndarray)
    assert current_state.shape == (3, 3)

    # get available actions
    game_state = current_state.copy()
    actions = get_available_actions(game_state)

    # First check for immediate winning moves
    for action in actions:
        temp_state = action_result(game_state.copy(), action, 1)
        judge = GameJudge()
        judge.game_state = temp_state
        if judge.check_game_status() == 1:
            return action
    
    # Then check for blocking opponent's winning moves
    for action in actions:
        temp_state = action_result(game_state.copy(), action, -1)
        judge = GameJudge()
        judge.game_state = temp_state
        if judge.check_game_status() == -1:
            return action
    
    # If no immediate wins or blocks, use minimax search
    values = []
    depth = 3
    for action in actions:
        values.append(min_value(action_result(game_state.copy(), action, 1), depth))
    max_ind = int(np.argmax(values))
    row, col = actions[max_ind][0], actions[max_ind][1]

    return row, col


def get_available_actions(current_state):
    """
    get all the available actions given current state
    :param current_state: current state of the game, it's a 3x3 array
    :return: available actions. list of tuple [(r0, c0), (r1, c1), (r2, c2)]
    """
    assert isinstance(current_state, np.ndarray), 'current_state should be numpy ndarray'
    assert current_state.shape == (3, 3), 'current_state: expect 3x3 array, get {}'.format(current_state.shape)

    # Find all empty positions
    empty_positions = np.where(current_state == 0)
    actions = list(zip(empty_positions[0], empty_positions[1]))
    
    return actions


def action_result(current_state, action, player):
    """
    update the game state given the input action and player
    :param current_state: current state of the game, it's a 3x3 array
    :param action: current action, tuple type
    :param player: 1 for computer, -1 for human
    :return: new game state after the action
    """
    assert isinstance(current_state, np.ndarray), 'current_state should be numpy ndarray'
    assert current_state.shape == (3, 3), 'current_state: expect 3x3 array, get {}'.format(current_state.shape)
    assert player in [1, -1], 'player should be either 1(computer) or -1(you)'

    # Create a copy of the current state
    new_state = current_state.copy()
    
    # Apply the action
    row, col = action
    new_state[row, col] = player
    
    return new_state


def min_value(current_state, depth):
    """
    recursively call min_value and max_value, min_value is for human player(-1)
    :param current_state: current game state (3x3 numpy array)
    :param depth: remaining search depth
    :return: minimum utility value
    """
    # Check if game is over or depth limit reached
    judge = GameJudge()
    judge.game_state = current_state.copy()
    status = judge.check_game_status()
    if status != 2 or depth == 0:
        return utility(current_state, -1)
    
    # Get available actions
    actions = get_available_actions(current_state)
    
    # Initialize minimum value
    min_val = float('inf')
    
    # Evaluate each possible action
    for action in actions:
        # Get resulting state
        result_state = action_result(current_state.copy(), action, -1)
        
        # Recursively call max_value
        val = max_value(result_state, depth - 1)
        
        # Update minimum value
        if val < min_val:
            min_val = val
    
    return min_val


def max_value(current_state, depth):
    """
    recursively call min_value and max_value, max_value is for computer(1)
    :param current_state: current game state (3x3 numpy array)
    :param depth: remaining search depth
    :return: maximum utility value
    """
    # Check if game is over or depth limit reached
    judge = GameJudge()
    judge.game_state = current_state.copy()
    status = judge.check_game_status()
    if status != 2 or depth == 0:
        return utility(current_state, 1)
    
    # Get available actions
    actions = get_available_actions(current_state)
    
    # Initialize maximum value
    max_val = float('-inf')
    
    # Evaluate each possible action
    for action in actions:
        # Get resulting state
        result_state = action_result(current_state.copy(), action, 1)
        
        # Recursively call min_value
        val = min_value(result_state, depth - 1)
        
        # Update maximum value
        if val > max_val:
            max_val = val
    
    return max_val


def utility(current_state, flag):
    """
    return utility function given current state and flag
    :param current_state: current game state (3x3 numpy array)
    :param flag: 1 for computer (MAX), -1 for human (MIN)
    :return: utility value
    """
    # Check if game is over
    judge = GameJudge()
    judge.game_state = current_state.copy()
    status = judge.check_game_status()
    
    if status == 1:  # Computer wins
        return 1000 if flag == 1 else -1000
    elif status == -1:  # Human wins
        return -1000 if flag == 1 else 1000
    elif status == 0:  # Draw
        return 0
    
    # Evaluate board position
    score = 0
    
    # Evaluate rows and columns
    for i in range(3):
        row = current_state[i, :]
        col = current_state[:, i]
        score += evaluate_line(row, flag)
        score += evaluate_line(col, flag)
    
    # Evaluate diagonals
    diag1 = current_state[0, 0] + current_state[1, 1] + current_state[2, 2]
    diag2 = current_state[0, 2] + current_state[1, 1] + current_state[2, 0]
    score += evaluate_line(diag1, flag)
    score += evaluate_line(diag2, flag)
    
    # Center control
    if current_state[1, 1] == 1:
        score += 10 if flag == 1 else -10
    
    return score

def evaluate_line(line, flag):
    """
    Evaluate a single line (row, column or diagonal)
    :param line: numpy array representing the line
    :param flag: 1 for computer (MAX), -1 for human (MIN)
    :return: evaluation score for the line
    """
    if isinstance(line, int):  # For diagonals passed as sums
        if line == 2:
            return 100 if flag == 1 else -100
        elif line == -2:
            return -100 if flag == 1 else 100
        return 0
    
    count_computer = np.count_nonzero(line == 1)
    count_human = np.count_nonzero(line == -1)
    
    # Higher priority for blocking opponent's immediate threats
    if count_human == 2 and count_computer == 0:
        return -200 if flag == 1 else 200
    elif count_computer == 2 and count_human == 0:
        return 100 if flag == 1 else -100
    elif count_human == 1 and count_computer == 0:
        return -50 if flag == 1 else 50
    elif count_computer == 1 and count_human == 0:
        return 20 if flag == 1 else -20
    
    return 0


# Do not modify the following code
class GameJudge(object):
    def __init__(self):
        self.game_state = np.zeros(shape=(3, 3), dtype=int)

    def make_one_move(self, row, col, player):
        """
        make one move forward
        :param row: row index of the circle(cross)
        :param col: column index of the circle(cross)
        :param player: player = 1 for computer / player = -1 for human
        :return:
        """
        # 1 stands for circle, -1 stands for cross, 0 stands for empty
        assert 0 <= row <= 2, "row index of the move should lie in [0, 2]"
        assert 0 <= col <= 2, "column index of the move should lie in [0, 2]"
        assert player in [-1, 1], "player should be noted as -1(human) or 1(computer)"
        self.game_state[row, col] = player

    def check_game_status(self):
        """
        return game status
        :return: 1 for computer wins, -1 for human wins, 0 for draw, 2 in the play
        """

        # somebody wins
        sum_rows = np.sum(self.game_state, axis=1).tolist()
        if 3 in sum_rows:
            return 1
        if -3 in sum_rows:
            return -1

        sum_cols = np.sum(self.game_state, axis=0).tolist()
        if 3 in sum_cols:
            return 1
        if -3 in sum_cols:
            return -1

        sum_diag = self.game_state[0][0] + self.game_state[1][1] + self.game_state[2][2]
        if sum_diag == 3:
            return 1
        if sum_diag == -3:
            return -1

        sum_rdiag = self.game_state[0][2] + self.game_state[1][1] + self.game_state[2][0]
        if sum_rdiag == 3:
            return 1
        if sum_rdiag == -3:
            return -1

        # draw
        if len(np.where(self.game_state == 0)[0]) == 0:
            return 0

        # in the play
        return 2

    def human_input(self):
        """
        take the human's move in
        :return: row and column index of human input
        """
        print("Input the row and column index of your move")
        print("1, 0 means draw a cross on the row 1, col 0")
        ind, succ = None, False
        while not succ:
            ind = list(map(int, input().strip().split(',')))
            if ind[0] < 0 or ind[0] > 2 or ind[1] < 0 or ind[1] > 2:
                succ = False
                print("Invalid input, the two numbers should lie in [0, 2]")
            elif self.game_state[ind[0], ind[1]] != 0:
                succ = False
                print(" You can not put cross on places already occupied")
            else:
                succ = True
        return ind[0], ind[1]

    def print_status(self, player, status):
        """
        print the game status
        :param player: player of the last move
        :param status: game status
        :return:
        """
        print("-----------------------------------------------------")
        for row in range(3):
            for col in range(3):
                if self.game_state[row, col] == 1:
                    print("[O]", end="")
                elif self.game_state[row, col] == -1:
                    print("[X]", end="")
                else:
                    print("[ ]", end="")
            print("")

        if player == 1:
            print("Last move was conducted by computer")
        elif player == -1:
            print("Last move was conducted by you")

        if status == 1:
            print("Computer wins")
        elif status == -1:
            print("You win")
        elif status == 2:
            print("Game going on")
        elif status == 0:
            print("Draw")

    def get_game_state(self):
        return self.game_state
