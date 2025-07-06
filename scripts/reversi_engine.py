# Constants
EMPTY, SOLID, HOLLOW, HINT = 0, 1, 2, 3
SIZE = 8
DIRECTIONS = [
    (-1, -1), (0, -1), (1, -1),
    (-1, 0),           (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

# Board state
def new_board():
    board = [[EMPTY for _ in range(SIZE)] for _ in range(SIZE)]
    board[3][3] = HOLLOW
    board[3][4] = SOLID
    board[4][3] = SOLID
    board[4][4] = HOLLOW
    return board

def init_game_state():
    initial_state = {
        "board": new_board(),
        "current_player": SOLID
    }   
    save_game_state(initial_state)

def get_board_hash(board):
    import hashlib
    import json
    board_str = json.dumps(board, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(board_str.encode()).hexdigest()[:8]

def load_game_state(path="game_state.json"):
    import json
    with open(path, 'r') as file:
        return json.load(file)

def save_game_state(game_state, path="game_state.json"):
    import json
    with open(path, 'w') as file:
        json.dump(game_state, file, sort_keys=True, separators=(",", ":"))

def is_game_over(board):
    return not has_valid_move(board, SOLID) and not has_valid_move(board, HOLLOW)

# Move logic
def is_valid_move(board, x, y, player):
    if board[x][y] != EMPTY:
        return False
    opponent = get_opponent(player)
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        has_opponent = False
        while 0 <= nx < SIZE and 0 <= ny < SIZE:
            if board[nx][ny] == opponent:
                has_opponent = True
            elif board[nx][ny] == player and has_opponent:
                return True
            else:
                break
            nx += dx
            ny += dy
    return False


def get_valid_moves(board, player):
    return [(x, y) for x in range(SIZE) for y in range(SIZE) if is_valid_move(board, x, y, player)]

def apply_move(board, x, y, player):
    if not is_valid_move(board, x, y, player):
        return board
    board[x][y] = player
    opponent = get_opponent(player)
    flipped_count = 0
    for dx, dy in DIRECTIONS:
        nx, ny = x + dx, y + dy
        flipped = []
        while 0 <= nx < SIZE and 0 <= ny < SIZE:
            if board[nx][ny] == opponent:
                flipped.append((nx ,ny))
            elif board[nx][ny] == player:
                for pos_x, pos_y in flipped:
                    board[pos_x][pos_y] = player
                flipped_count += len(flipped)
                break
            else:
                break
            nx += dx
            ny += dy
    return board, flipped_count

def has_valid_move(board, player):
    return len(get_valid_moves(board, player)) > 0

# Helpers
def parse_coord(coord_str):
    col = ord(coord_str[0].upper()) - ord('A')
    row = int(coord_str[1]) - 1
    return row, col

def coord_str(x, y):
    return f"{chr(ord('A') + y)}{x + 1}"

def get_opponent(player):
    return SOLID if player == HOLLOW else HOLLOW

def plot_hints(board, player):
    from copy import deepcopy
    board_with_hints = deepcopy(board)
    for x in range(SIZE):
        for y in range(SIZE):
            if is_valid_move(board_with_hints, x, y, player):
                board_with_hints[x][y] = HINT
    return board_with_hints

def count_score(board):
    solid = sum(row.count(SOLID) for row in board)
    hollow = sum(row.count(HOLLOW) for row in board)
    return solid, hollow

# Records
def record_flip(user, flipped, path="flip_history.jsonl"):
    from datetime import datetime, timezone
    import json
    entry = {
        "user": user,
        "flipped": flipped,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_leaderboard(path="flip_history.jsonl"):
    import os, json
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]
    
def get_top_flips(history, top_n=5):
    return sorted(history, key=lambda x: x['flipped'], reverse=True)[:top_n]

def get_top_contributors(history, top_n=5):
    from collections import Counter
    counter = Counter(entry['user'] for entry in history)
    return counter.most_common(top_n)