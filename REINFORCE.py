import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np

#------hyperparameters------
discount_factor = 0.99


class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, state):
        return Categorical(logits=self.network(state))

policy = PolicyNetwork()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
env = gym.make("CartPole-v1", render_mode = 'human')  # initializing the environment

for episode in range(10000):
    log_probs = []
    rewards = []

    observation, _ = env.reset()

    for t in range(500):
        state = torch.tensor(observation, dtype=torch.float32)  # assuming the observation = state here
        dist = policy(state)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        observation, reward, terminated, truncated, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(reward)

        if terminated or truncated:
            break

    # --- episode has ended here, this block runs ONCE per episode (indented inside "for episode", outside "for t") ---

    # computing reward-to-go, ONE VALUE PER TIMESTEP, not one aggregate number
    returns = []
    G = 0
    for r in reversed(rewards):        # walk backwards through the episode
        G = r + discount_factor * G    # G_t = r_t + γ * G_{t+1}
        returns.insert(0, G)           # put it at the front so order matches log_probs/rewards

    returns = torch.tensor(returns, dtype=torch.float32)

    # normalise (this is your baseline subtraction, reduces variance)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # --- compute loss and update ---
    loss = []
    for log_prob, G in zip(log_probs, returns):
        loss.append(-log_prob * G)     # negative because PyTorch minimises, we want to maximise
    loss = torch.stack(loss).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 100 == 0:
        print(f"Episode {episode}, total reward: {sum(rewards)}")