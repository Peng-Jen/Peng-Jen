import json
import numpy as np
from reversi_engine import (
    load_game_state, apply_move, get_valid_moves, is_game_over, coord_str, flatten_board_indices,
    get_opponent, has_valid_move, save_game_state, plot_hints, load_leaderboard, encode_board
)
from render_board import write_readme
from dqn_agent import DQNAgent

REPO_URL = "https://github.com/Peng-Jen/Peng-Jen"

if __name__ == "__main__":
    game_state = load_game_state("game_state.json")
    board = game_state["board"]
    current_player = game_state["current_player"]

    if is_game_over(board):
        print("Game over. No AI move.")
        exit(0)

    agent = DQNAgent(model_path="models/dqn.pt")
    valid_moves = get_valid_moves(board, current_player)
    flattened_indices = flatten_board_indices(valid_moves)

    if not valid_moves:
        print("No valid moves for AI.")
        with open("ai_move_log.json", "w") as file:
            json.dump({"move": None, "pass": True}, file)
        exit(0)

    encoded = encode_board(board)
    action = agent.act(encoded, flattened_indices)
    x, y = divmod(action, 8)

    if (x, y) not in valid_moves:
        print(f"Invalid AI move: {(x, y)} not in {valid_moves}")
        exit(1)

    with open("ai_move_log.json", "w") as file:
        json.dump({
            "move": coord_str(x, y),
            "pass": False
        }, file)

    board, flips = apply_move(board, x, y, current_player)
    next_player = get_opponent(current_player)

    if not has_valid_move(board, next_player) and has_valid_move(board, current_player):
        next_player = current_player

    game_state["board"] = board
    game_state["current_player"] = next_player
    save_game_state(game_state)

    board_with_hints = plot_hints(board, next_player)
    history = load_leaderboard()
    write_readme(board_with_hints, board, next_player, REPO_URL, history)

    print(f"AI moved: {coord_str(x, y)} flips: {flips}")
