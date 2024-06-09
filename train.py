import argparse
import wandb
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epsilon_decay', type=float, default=0.995)
parser.add_argument('--learning_rate', type=float, default=0.001)
args = parser.parse_args()

# Initialize wandb
wandb.init(project='wandb_testing', config=args)
config = wandb.config

print('init done')

# Define the CartPole-v1 environment
env = gym.make('CartPole-v1')
print('cartpole made')

# Hyperparameters
gamma = 0.99  # Discount factor
epsilon_start = 1.0  # Initial exploration rate
epsilon_end = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate of exploration
learning_rate = 0.001  # Learning rate for optimizer
batch_size = 64  # Batch size for experience replay
memory_size = 10000  # Maximum size of replay memory
n_episodes = 100  # Number of episodes for training
target_update = 10  # Frequency of updating the target network
print('Hyperparameters made')

# Define the Q-network using PyTorch
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Experience replay memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Initialize environment and Q-network
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = QNetwork(state_size, action_size)
target_net = QNetwork(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = ReplayMemory(memory_size)

# Epsilon-greedy policy for action selection
epsilon = epsilon_start

def select_action(state):
    global epsilon
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_net(state).argmax().view(1, 1)
    else:
        return torch.tensor([[random.randrange(action_size)]], dtype=torch.long)

# Training loop
for episode in range(n_episodes):
    print('in training loop!')
    state, _ = env.reset()  # Extract the state array from the tuple
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    total_reward = 0  # Initialize total reward for the episode

    for t in range(200):  # Run for a maximum of 200 steps
        action = select_action(state)
        result = env.step(action.item())
        next_state = result[0]
        reward = result[1]
        done = result[2]

        reward = torch.tensor([reward], dtype=torch.float32)
        total_reward += reward.item()

        if not done:
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        if len(memory) > batch_size:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = policy_net(state_batch).gather(1, action_batch)

            next_state_values = torch.zeros(batch_size)
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
            expected_state_action_values = (next_state_values * gamma) + reward_batch

            loss = nn.functional.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Print statements for debugging and tracking progress
    print(f"Episode {episode + 1}/{n_episodes}")
    print(f"Total Reward: {total_reward}")
    if len(memory) > batch_size:
        print(f"Loss: {loss.item()}")
    print(f"Epsilon: {epsilon}\n")

    # Log metrics to wandb
    wandb.log({"episode": episode, "reward": total_reward, "loss": loss.item() if len(memory) > batch_size else None, "epsilon": epsilon})

env.close()
wandb.finish()
