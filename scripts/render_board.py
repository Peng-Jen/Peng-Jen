import os
from reversi_engine import SOLID, HOLLOW, EMPTY, HINT, SIZE
from reversi_engine import (
    coord_str, get_board_hash, is_game_over, count_score,
    get_top_flips, get_top_contributors
)

REPO_URL = "https://github.com/Peng-Jen/Peng-Jen"

REVERSI = """\
# Github Community's Reversi

## What's Reversi
Reversi is a turn-based strategy board game for two players: Black (Solid) and White (Hollow). The game is played on an 8x8 board.
- Players take turns placing a disc of their color on the board.
- A valid move must capture one or more of the opponent's discs by "sandwiching" them in a straight line (horizontal, vertical, or diagonal).
- All sandwiched discs are flipped to the current player's color.
- If a player has no valid move, they pass the turn.
- The game ends when neither player can move. The player with more discs wins.

## How to play
- Click a hint dot to open an issue.
- Simply click "Create" to play that move.
- Github Actions workflow will update my profile README.md.
- You'll see the board updated shortly.

## Current state
This is the **AI Challenge Version**  
Click a move to play against the bot.  
Prefer human vs. human? [Switch to community version](https://github.com/Peng-Jen/Peng-Jen/tree/Community)

Now, it's your turn!
"""

AGENT = """\
# About Agent

The default AI agent is a Deep Q-Network trained on the 8x8 Reversi board.  
It uses a convolutional neural network to estimate Q-values for each of the 64 board positions and selects actions using an epsilon-greedy policy.

## Training Configuration:
- **Network**: Simple CNN with linear head
- **Input**: 8x8x3 board state tensor
- **Output**: Q-values for all 64 positions
- **Loss Function**: SmoothL1Loss (Huber)
- **Optimizer**: Adam, learning rate = 1e-5
- **Discount Factor ($\gamma$)**: 0.99
- **Exploration**: Epsilon-greedy  
- **Replay Buffer Size**: 100,000
- **Batch Size**: 128
- **Target Network Update**: Soft update with $\\tau$ = 0.01

## Training Progress:

The following plot shows the agent's performance over 10,000 episodes:
- **Top**: Episode reward (with rolling average)
- **Middle**: Loss during updates
- **Bottom**: Average Q-value across actions

<div style="text-align:center;">
  <img src="training_images/training_plot.png" width="640" height="640" />
</div>

## Disclaimer

This project is my personal exploration into reinforcement learning and game AI.  
I'm still quite new to the field of RL, so the current agent, reward design, and training pipeline might not reflect best practices.

If you have any feedback, suggestions, or ideas for improvement — whether it's about the model architecture, training strategies, or gameplay logic — I'd love to hear from you!

Feel free to open an issue, submit a pull request, or just reach out.  
Any guidance from experienced practitioners is genuinely appreciated!
"""

INTRO = """\
# About me
## Tim Chen #Peng-Jen
- Studying in Communication Engineering @ National Taiwan University
- International Exchange Student @ Keio University, Japan
- B.S. in Mathematics & B.B.A. in Information Management @ National Taiwan University

## Interested Area
- Operations Research, Optimization
- Decision Analysis
- Data Science
- Reinforcement Learning

## Skill Set
- Operations Research
- Machine Learning
- Applied Mathematics

## Learning Topics
- Game Theory
- Discrete Optimization Algorithm

## Connect with me
- [LinkedIn](https://www.linkedin.com/in/tim-chen-1a92a825a/)
- tim.pjchen@gmail.com
"""

IMG_PATH = "./reversi_images"

IMG_MAP = {
    EMPTY: "blank.png",
    SOLID: "solid.png",
    HOLLOW: "hollow.png",
    HINT: "hint_dot.png"
}

def render_board_md(board, board_for_hash, repo_url):
    """
    Generate Markdown table of the board with clickable HINTs.
    board: the board with HINTS for display
    board_for_hash: the board used for computing hash (should NOT include HINTs)
    """
    header = "|   | " + " | ".join([chr(ord('A') + i) for i in range(SIZE)]) + " |"
    divider = "|---" + "|---" * SIZE + "|"
    rows = [header, divider]

    board_hash = get_board_hash(board_for_hash)

    for i in range(SIZE):
        row_cells = [f"**{i + 1}**"]
        for j in range(SIZE):
            img_file = IMG_MAP[board[i][j]]
            if board[i][j] == HINT:
                coord = coord_str(i, j)
                url = f"{repo_url}/issues/new?title=%5BAI%5D+move+{coord}+{board_hash}&body=Just+click+'Create'.+You+don't+need+to+do+anything+else."
                img_md = f"[![]({IMG_PATH}/hint_dot.png)]({url})"
            else:
                img_md = f"<img src='{IMG_PATH}/{img_file}' width='64' height='64'/>"
            row_cells.append(img_md)
        rows.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(rows)

def write_readme(board, board_for_hash, current_player, repo_url, history=None):
    """
    Write full README.md content.
    board: board with HINTs for rendering
    board_for_hash: raw board (without hints) for hash computation and game logic
    """
    board_md = render_board_md(board, board_for_hash, repo_url)

    with open("README.md", "w") as f:
        f.write(REVERSI)

        if is_game_over(board_for_hash):
            solid, hollow = count_score(board_for_hash)
            winner = "Player" if solid > hollow else "The Bot"
            f.write(board_md + "\n\n")
            f.write(f"### 🎉 Game Over (**{winner}** wins)\n")
            f.write(f"**Solid**: {solid} vs. **Hollow**: {hollow}\n\n")
            f.write(f"Click [here]({REPO_URL}/issues/new?title=%5BAI%5D+new_game) to start a new game\n")
        else:
            f.write(f"**{'SOLID' if current_player == SOLID else 'HOLLOW'}** to move.\n\n")
            f.write(board_md + "\n\n")
        
        if history:
            f.write("\n")
            f.write(render_leaderboard(history) + "\n\n")
            f.write(render_contributors(history) + "\n\n")

        f.write(AGENT)
        f.write(INTRO)

def render_leaderboard(history):
    top_flips = get_top_flips(history)
    if not top_flips:
        return "### Best Single Moves\n_No data yet._"
    rows = ["### Best Single Moves", "| Rank | User | Flipped |", "|------|------|---------|"]
    for i, entry in enumerate(top_flips):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
        rows.append(f"| {medal} | [@{entry['user']}](https://github.com/{entry['user']}) | {entry['flipped']} |")
    return "\n".join(rows)

def render_contributors(history):
    top_users = get_top_contributors(history)
    if not top_users:
        return "### Top Contributors\n_No contributions yet._"
    rows = ["### Top Contributors", "| Rank | User | Moves |", "|------|------|--------|"]
    for i, (user, count) in enumerate(top_users):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else str(i + 1)
        rows.append(f"| {medal} | [@{user}](https://github.com/{user}) | {count} |")
    return "\n".join(rows)
