import sys
from reversi_engine import (
    init_game_state, get_board_hash, load_game_state, save_game_state, is_game_over,
    apply_move, get_valid_moves, has_valid_move, parse_coord, plot_hints, record_flip,
    load_leaderboard
)
from reversi_engine import SOLID, HOLLOW
from render_board import write_readme

REPO_URL = "https://github.com/Peng-Jen/Peng-Jen"

def get_next_player(current_player):
    return HOLLOW if current_player == SOLID else SOLID

def handle_new_game(path="game_state.json", force=False):
    # Only allow new game if game is over, or force is explicitly passed
    if not force:
        try:
            state = load_game_state(path)
            board = state["board"]
            if not is_game_over(board):
                print("Game is not over. Cannot start a new game.")
                return
        except Exception as e:
            print("Failed to load game state:", str(e))
            return

    init_game_state()
    state = load_game_state(path)
    board = state["board"]
    board_with_hints = plot_hints(board, SOLID)
    history = load_leaderboard()
    write_readme(board_with_hints, board, SOLID, REPO_URL, history)

def handle_move(coord_str, hash_from_issue, user, path="game_state.json"):
    state = load_game_state(path)
    board = state["board"]
    current_player = state["current_player"]

    # Check hash
    current_hash = get_board_hash(board)
    if current_hash != hash_from_issue:
        print("Outdated request")
        return  # Ignore outdated move

    # Parse and validate move
    x, y = parse_coord(coord_str)
    if (x, y) not in get_valid_moves(board, current_player):
        return

    board, flipped_counts = apply_move(board, x, y, current_player)
    
    record_flip(user, flipped_counts)

    next_player = get_next_player(current_player)

    # Check if next player can move; if not, same player continues
    if not has_valid_move(board, next_player) and has_valid_move(board, current_player):
        next_player = current_player

    # Save state and update board
    state["board"] = board
    state["current_player"] = next_player
    save_game_state(state, path)

    board_with_hints = plot_hints(board, next_player)
    history = load_leaderboard()
    write_readme(board_with_hints, board, next_player, REPO_URL, history)

if __name__ == "__main__":
    args = sys.argv[1:]

    # Match GitHub Actions CLI calls
    if len(args) >= 1 and args[0].lower() == "new_game":
        force = (len(args) == 2 and args[1].lower() == "true")
        handle_new_game(force=force)

    elif len(args) == 3:
        coord = args[0].upper()
        hash_val = args[1]
        user = args[2]
        handle_move(coord, hash_val, user)

    else:
        print("Usage:")
        print("  python handle_issue.py [Community|AI] new_game [true|false]")
        print("  python handle_issue.py [Community|AI] <COORD> <HASH>")
        sys.exit(1)
