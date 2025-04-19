"""
DQN is 
"""
# Standard library imports
import random
from collections import deque
import os

# Third party imports
import ale_py
import numpy as np
import gymnasium as gym
from sympy import false
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from gymnasium.vector import SyncVectorEnv

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    
with open("loss.txt", "w") as f:
    print(f"Using device: {device}")
    f.write(f"Using device: {device}\n")

os.makedirs("models", exist_ok=True)

# Initialise the Farama Gym environment
def make_env():
    env = gym.make("BreakoutDeterministic-v4", frameskip=1)
    # Set up gameplay to skip 4 frames so game feels sped up for model
    # Set up screen size as a square to avoid padding requirements
    # Set grayscale to reduce channel dimensions with 4 frames to demonstrate velocity
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
    # Stack the frames so that each observation of the environment state contains 4 frames
    # Note that in tandem with the above, the model sees 16 frames for every 1 frame in real time
    env = FrameStackObservation(env, stack_size=4)
    return env

num_envs = 4  # Number of parallel environments
env = SyncVectorEnv([make_env for _ in range(num_envs)])

num_actions = 4

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Recall: output_size = floor((input_size - kernel_size + 2*padding) / stride) + 1
        # Input shape: batch * 4 * 84 * 84 (batch * frame_stack_size * height * width)
        self.conv1 = nn.Conv2d(4, 32, 8, 2)  # Output shape: batch * 32 * 39 * 39
        # Input shape: batch * 32 * 39 * 39
        self.conv2 = nn.Conv2d(32, 64, 4, 2)  # Output shape: batch * 64 * 18 * 18
        # Input shape: batch * 64 * 18 * 18
        self.conv3 = nn.Conv2d(64, 64, 3, 2)  # Output shape: batch * 64 * 8 * 8
        # Input shape: batch * (64x8x8) (after flattening)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Output shape: batch * 512
        self.fc2 = nn.Linear(512, num_actions)  # Output shape: batch * num_actions
        
    def forward(self, input) -> torch.Tensor:
        # input shape: batch * 4 * 84 * 84
        x = torch.relu(self.conv1(input)) # Output shape: batch * 32 * 39 * 39
        x = torch.relu(self.conv2(x)) # Output shape: batch * 64 * 18 * 18
        x = torch.relu(self.conv3(x)) # Output shape: batch * 64 * 16 * 16
        # Flatten each 3D tensor of the batch into a 1D vector (-1 predicts dim required)
        # batch * 64 * 16 * 16 -> batch * (64x16x16 = 16384)
        x = x.view(x.size(0), -1) # Output shape: batch * (64x16x16 = 16384)
        x = torch.relu(self.fc1(x)) # Output shape: batch * 512
        x = self.fc2(x) # Output shape: batch * num_actions
        return x
        
# Main network that will be optimised at each batch
dqn = DQN(num_actions).to(device)

# A duplicate of an older DQN to stop heavy divergence and improve training stability
target_net = DQN(num_actions).to(device)
target_net.load_state_dict(dqn.state_dict())
target_net.eval()

# Stores the last 1M tuples of (state, action, reward, next_state, done) to sample
# from randomly during training and avoiding overfitting to recent events
replay_buffer = deque(maxlen=1000000)

# Outputs a number between 0-3 (4 possible actions)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()  # Sample actions for all environments
    else:
        # Turn into PyTorch tensors and then divide each by 255 to normalise each pixel to
        # between 0-1. Also unsqueeze to add a batch dimension at the start.
        # Input state shape: num_envs * 4 * 84 * 84
        state_tensor = torch.from_numpy(np.array(state)).float().to(device)  # Move to GPU
        state_tensor = state_tensor.div(255)  # Normalize
        # Outut state shape: num_envs * 4 * 84 * 84 (with normalised pixels)
        with torch.no_grad():
            q_vals = dqn(state_tensor)  # Output shape: num_envs * num_actions
        # Pick the action with the highest quality value for each environment
        return q_vals.argmax(dim=1).cpu().numpy()  # Output shape: num_envs

num_episodes = 50000
batch_size = 128
discount_factor = 0.99
epsilon_min = 0.01
epsilon_start = 1.0
total_steps = 0
target_net_update_freq = 5000
epsilon_decay_steps = 50000

# Train only the DQN and target network will simply copy the state dict occasionally
optimiser = optim.Adam(dqn.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

for episode in range(num_episodes):
    # Outputs state as LazyFrames
    state, info = env.reset()  # state shape: num_envs * 4 * 84 * 84
    # LazyFrames -> np arrays
    state = np.array(state)
    episode_reward = np.zeros(num_envs)  # Track rewards for each environment
    step = 0
    is_done = np.zeros(num_envs, dtype=bool)  # Track done flags for each environment
    
    while not is_done.all():  # Continue until all environments are done
        epsilon = max(
            epsilon_min, 
            epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay_steps)
        )
        
        action = choose_action(state, epsilon)  # Output shape: num_envs
        next_state, reward, is_done, _, _ = env.step(action)  # All outputs are arrays
        next_state = np.array(next_state)  # Output shape: num_envs * 4 * 84 * 84
        reward = np.array(reward)  # Output shape: num_envs
        reward = np.clip(reward, -1, 1)
        
        # Add tuples of the current step's state for each environment
        for i in range(num_envs):
            replay_buffer.append((state[i], action[i], reward[i], next_state[i], is_done[i]))
        
        state = next_state
        episode_reward += reward
        
        if len(replay_buffer) > batch_size:
            # Sample a set of 32 random events and actions taken
            # Use NumPy for faster sampling
            indices = np.random.choice(len(replay_buffer), batch_size, replace=False)
            batch = [replay_buffer[i] for i in indices]
            states, actions, rewards, next_states, is_dones = zip(*batch)
            
            # States shape: batch * 4 * 84 * 84
            states = np.array(states)  # Convert tuple of lists to NumPy array
            states = torch.from_numpy(states).float().div(255).to(device)
            # Actions shape: batch * 1
            actions = torch.tensor(actions).unsqueeze(1).to(device)  # Add an extra dimension
            # Rewards shape: batch * 1
            rewards = torch.tensor(rewards).float().to(device)
            # Next states shape: batch * 4 * 84 * 84
            next_states = np.array(next_states)  # Convert tuple of lists to NumPy array
            next_states = torch.from_numpy(next_states).float().div(255).to(device)
            # Is dones shape: batch * 1
            is_dones = torch.tensor(is_dones).float().to(device)
            
            # Output of DQN has shape: batch * num_actions
            q_values = dqn(states).gather(1, actions).squeeze()  # Now actions is 2D
            
            with torch.no_grad():
                # Finds the highest possible next q value for the next state
                max_next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + discount_factor * max_next_q_values * (1 - is_dones)
                
            loss = loss_fn(q_values, target_q_values)
            optimiser.zero_grad()
            if step % 20 == 0:
                # print(f"episode: {episode}, step: {step}, loss: {loss.item()}")
                with open("loss.txt", "a") as f:
                    f.write(f"episode: {episode}, step: {step}, loss: {loss.item()}\n")
            loss.backward()
            optimiser.step()
        
        if step % target_net_update_freq == 0:
            target_net.load_state_dict(dqn.state_dict())
            
        step += 1
        total_steps += 1
        
        if is_done.all():
            break
            
    print(f"Episode {episode} finished with reward {episode_reward.sum()}")
    with open("loss.txt", "a") as f:
        f.write(f"Episode {episode} finished with reward {episode_reward.sum()}\n")
    
    if episode % 500 == 0:
        # Save models
        torch.save(dqn.state_dict(), f"models/dqn_{episode}.pth")
        torch.save(target_net.state_dict(), f"models/target_net_{episode}.pth")
        
torch.save(dqn.state_dict(), f"models/dqn_latest.pth")
torch.save(target_net.state_dict(), f"models/target_net_latest.pth")
