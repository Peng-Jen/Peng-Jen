import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 64)
        )

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.fc(x)
        return x


class DQNAgent:
    def __init__(self, model_path="models/dqn.pt", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def act(self, state: np.ndarray, legal_moves: list[int]) -> int:
        """
        Args:
            state: np.ndarray of shape (8, 8, 3)
            legal_moves: list of legal action indices (0~63)

        Returns:
            selected_action: int (0~63)
        """
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)  # (1, 3, 8, 8)
            q_values = self.model(x)[0].cpu().numpy()  # (64,)

        # mask illegal moves to -inf
        mask = np.full_like(q_values, -np.inf)
        mask[legal_moves] = q_values[legal_moves]
        return int(np.argmax(mask))
