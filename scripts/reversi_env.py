import numpy as np
from reversi_engine import (
    get_valid_moves, is_game_over, apply_move, get_opponent, count_score,
    SOLID, HOLLOW
)
from copy import deepcopy


class ReversiEnv():
    def __init__(self, stable_reward=False):
        self.stable_reward = stable_reward
        self.reset()

    def reset(self):
        self.size = 8
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.board[3, 3], self.board[4, 4] = 2, 2
        self.board[3, 4], self.board[4, 3] = 1, 1
        self.current_player = 1
        return self._get_state()

    def _get_state(self):
        p1 = (self.board == self.current_player).astype(np.float32)
        p2 = ((self.board != 0) & (self.board != self.current_player)).astype(np.float32)
        legal = np.zeros_like(p1)
        for r, c in get_valid_moves(self.board, self.current_player):
            legal[r, c] = 1.0
        return np.stack([p1, p2, legal], axis=-1)

    def count_stable_corners(self, board, player):
        stable = 0
        boundary = self.size - 1
        corners = [(0, 0), (0, boundary), (boundary, 0), (boundary, boundary)]
        for r, c in corners:
            if board[r, c] == player:
                stable += 1
        return stable

    def step(self, action):
        r, c = divmod(action, self.size)
        if (r, c) not in get_valid_moves(self.board, self.current_player):
            return self._get_state(), -10, True, {}

        pre_stable = self.count_stable_corners(self.board, self.current_player) if self.stable_reward else 0
        self.board, flipped = apply_move(self.board, r, c, self.current_player)
        post_stable = self.count_stable_corners(self.board, self.current_player) if self.stable_reward else 0

        reward = flipped * 0.1
        if self.stable_reward:
            reward += (post_stable - pre_stable) * 0.5  # Bonus per new stable corner

        opponent = get_opponent(self.current_player)
        opponent_moves = get_valid_moves(self.board, opponent)

        max_flips = 0
        for ox, oy in opponent_moves:
            temp_board, flips = apply_move(deepcopy(self.board), ox, oy, opponent)
            max_flips = max(max_flips, flips)

        penalty = max_flips * 0.05
        reward -= penalty


        if is_game_over(self.board):
            solid_score, hollow_score = count_score(self.board)
            winner = SOLID if solid_score > hollow_score else HOLLOW
            score_diff = solid_score - hollow_score if winner == SOLID else hollow_score - solid_score
            if winner == self.current_player:
                reward += 5 + score_diff * 0.05
            else:
                reward -= 5 + score_diff * 0.05
            done = True
        else:
            done = False
            self.current_player = opponent

        return self._get_state(), reward, done, {}

    def render(self):
        symbols = ['.', '●', '○', '*']
        for r in range(self.size):
            print(' '.join(symbols[x] for x in self.board[r]))
        print()

    def legal_action_indices(self):
        moves = get_valid_moves(self.board, self.current_player)
        return [self.size * r + c for r, c in moves]
