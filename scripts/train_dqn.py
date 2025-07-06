import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from reversi_env import ReversiEnv
from dqn_agent import DQN
import os
from tqdm import tqdm

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 10000
TAU = 0.01  # For soft update
NUM_EPISODES = 50000
REPLAY_BUFFER_SIZE = 100000
MIN_REPLAY_SIZE = 3000
SAVE_PATH = "models/dqn.pt"
PLOT_PATH = "training_images/training_plot.png"

losses = []
avg_q_values = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def epsilon_by_frame(frame_idx):
    return EPS_END + (EPS_START - EPS_END) * np.exp(-1. * frame_idx / EPS_DECAY)


def compute_loss(batch, policy_net, target_net):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states)
    next_q_values = target_net(next_states)

    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    max_next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + GAMMA * max_next_q_value * (1 - dones)

    loss = nn.SmoothL1Loss()(q_value, expected_q_value.detach())
    return loss, q_value.mean().item()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def sample_batch(replay_buffer):
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)
    return np.array(states), actions, rewards, np.array(next_states), dones


def plot_metrics(rewards, losses, q_values, path):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    axs[0].plot(rewards, label='Episode Reward')
    if len(rewards) >= 100:
        rolling = np.convolve(rewards, np.ones(100)/100, mode='valid')
        axs[0].plot(range(99, len(rewards)), rolling, label='Avg(100)', linestyle='--')
    axs[0].set_title('Rewards')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(losses, label='Training Loss')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Update Step')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(q_values, label='Avg Q-Value')
    axs[2].set_title('Q Values')
    axs[2].set_xlabel('Update Step')
    axs[2].set_ylabel('Q Value')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()


def main():
    env = ReversiEnv(stable_reward=True)
    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=5e-5)
    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)

    frame_idx = 0
    all_rewards = []

    os.makedirs("models", exist_ok=True)

    for episode in tqdm(range(NUM_EPISODES)):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            legal_moves = env.legal_action_indices()
            if not legal_moves:
                break

            epsilon = epsilon_by_frame(frame_idx)
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                q_values = policy_net(state_tensor)[0].detach().cpu().numpy()
                q_values_masked = np.full_like(q_values, -np.inf)
                q_values_masked[legal_moves] = q_values[legal_moves]
                action = int(np.argmax(q_values_masked))

            next_state, reward, done, _ = env.step(action)
            replay_buffer.append((np.copy(state), action, reward, np.copy(next_state), done))
            state = next_state
            episode_reward += reward
            frame_idx += 1

            if len(replay_buffer) > MIN_REPLAY_SIZE:
                batch = sample_batch(replay_buffer)
                loss, avg_q = compute_loss(batch, policy_net, target_net)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

                losses.append(loss.item())
                avg_q_values.append(avg_q)

                soft_update(target_net, policy_net, TAU)

        all_rewards.append(episode_reward)
        if episode % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, Avg(100): {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

        if episode % 500 == 0:
            torch.save(policy_net.state_dict(), SAVE_PATH)
            print(f"Model saved to {SAVE_PATH}")
            plot_metrics(all_rewards, losses, avg_q_values, PLOT_PATH)

    torch.save(policy_net.state_dict(), SAVE_PATH)
    plot_metrics(all_rewards, losses, avg_q_values, PLOT_PATH)
    print("Final model saved.")


if __name__ == "__main__":
    main()
