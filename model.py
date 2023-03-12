import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class LinearQNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(LinearQNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save_model(self, save_dir="./", file_name="model.pth"):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, file_name)
        torch.save(self.action_dict(), save_path)


class QTrainer:
    def __init__(self, model, learning_rate, gamma) -> None:
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over, )

        prediction = self.model(state)
        target = prediction.clone()

        for idx in range(len(game_over)):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new += self.gamma * torch.max(self.model(new_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        self.optimizer.step()
