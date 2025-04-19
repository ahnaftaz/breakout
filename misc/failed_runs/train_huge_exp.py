"""
DQN implementation for Atari games
"""
# Standard library imports
import random
from collections import deque
import os
import time
import datetime

# Third party imports
import ale_py
import numpy as np
import gymnasium as gym
from sympy import false
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

import wandb

# Create a timestamp for the run
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"dqn_run_{timestamp}"

# Create directories for logs and models
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs(f"models/{run_name}", exist_ok=True)

# Setup logging files
log_file = f"logs/loss_{run_name}.txt"
progress_file = f"logs/progress_{run_name}.txt"

# Initialize wandb
wandb.init(
    # Set the wandb entity where your project will be logged
    entity="ahnaft-dev",
    # Set the wandb project where this run will be logged
    project="sudoku",
    # Name this run
    name=run_name,
    # Track hyperparameters and run metadata
    save_code=True,
)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    
with open(log_file, "w") as f:
    print(f"Using device: {device}")
    f.write(f"Using device: {device}\n")

with open(progress_file, "w") as f:
    f.write(f"Run started at {timestamp} using {device}\n")

# Initialise the Farama Gym environment
env = gym.make("BreakoutDeterministic-v4", frameskip=1)

# Set up gameplay to skip 4 frames so game feels sped up for model
# Set up screen size as a square to avoid padding requirements
# Set grayscale to reduce channel dimensions with 4 frames to demonstrate velocity
env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)

# Stack the frames so that each observation of the environment state contains 4 frames
# Note that in tandem with the above, the model sees 16 frames for every 1 frame in real time
env = FrameStackObservation(env, stack_size=4)

num_actions = env.action_space.n # type: ignore

# Define hyperparameters
hyperparams = {
    "batch_size": 8192,  # Using large batch size to leverage GPU memory
    "discount_factor": 0.99,
    "epsilon_min": 0.01,
    "epsilon_start": 1.0,
    "epsilon_decay_steps": 1000000,
    "learning_rate": 1e-4,
    "target_net_update_freq": 10000,
    "replay_buffer_size": 1000000,
    "num_episodes": 50000,
    "train_frequency": 1,  # Train every step
    "prefill_steps": 10000
}

# Log hyperparameters to wandb
wandb.config.update(hyperparams)

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
replay_buffer = deque(maxlen=hyperparams["replay_buffer_size"])

# Outputs a number between 0-3 (4 possible actions)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        # Turn into PyTorch tensors and then divide each by 255 to normalise each pixel to
        # between 0-1. Also unsqueeze to add a batch dimension at the start.
        # Input state shape: 4 * 84 * 84
        state_tensor = torch.from_numpy(np.array(state)).to(device)
        state_tensor = state_tensor.float().div(255).unsqueeze(0)
        # Outut state shape: 1 * 4 * 84 * 84 (with normalised pixels)
        with torch.no_grad():
            q_vals = dqn(state_tensor) # Output shape: 1 * 4
        # Pick the action with the highest quality value
        return q_vals.argmax().item() # Output shape: 1 (single val extracted)

# Setup training
batch_size = hyperparams["batch_size"]
discount_factor = hyperparams["discount_factor"]
epsilon_min = hyperparams["epsilon_min"]
epsilon_start = hyperparams["epsilon_start"]
total_steps = 0
target_net_update_freq = hyperparams["target_net_update_freq"]
epsilon_decay_steps = hyperparams["epsilon_decay_steps"]
train_frequency = hyperparams["train_frequency"]
prefill_steps = hyperparams["prefill_steps"]

# Train only the DQN and target network will simply copy the state dict occasionally
optimiser = optim.Adam(dqn.parameters(), lr=hyperparams["learning_rate"])
loss_fn = nn.MSELoss()
scaler = GradScaler()

# Log to both console and file
def log_message(message, to_progress=True, to_console=True):
    if to_console:
        print(message)
    if to_progress:
        with open(progress_file, "a") as f:
            f.write(message + "\n")

log_message("Pre-filling replay buffer...")
# Fill the replay buffer with random experiences before starting training
state, info = env.reset()
state = np.array(state)

for step in range(prefill_steps):
    if step % 1000 == 0:
        log_message(f"Prefilling step {step}/{prefill_steps}")
    action = env.action_space.sample()  # Take completely random actions during prefill
    next_state, reward, terminated, truncated, info = env.step(action)
    is_done = terminated or truncated
    next_state = np.array(next_state)
    reward = np.clip(reward, -1, 1) # type: ignore
    replay_buffer.append((state, action, reward, next_state, is_done))
    state = next_state
    if is_done:
        state, info = env.reset()
        state = np.array(state)
log_message(f"Replay buffer prefilled with {len(replay_buffer)} experiences")

# Training metrics
episode_rewards = []
episode_durations = []
episode_losses = []
train_times = []

for episode in range(hyperparams["num_episodes"]):
    # Outputs state as LazyFrames
    state, info = env.reset()
    # LazyFrames -> np arrays
    state = np.array(state) # Output shape: 4 * 84 * 84
    episode_reward = 0 
    step = 0
    is_done = False
    episode_start_time = time.time()
    episode_loss = []
    
    while not is_done:
        epsilon = max(
            epsilon_min, 
            epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay_steps)
        )
        
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        is_done = terminated or truncated
        next_state = np.array(next_state) # Output shape: 4 * 84 * 84
        reward = np.clip(reward, -1, 1) # type: ignore
        
        # Add a tuple of the current step's state
        replay_buffer.append((state, action, reward, next_state, is_done))
        state = next_state
        episode_reward += reward
        
        # Only train every few steps to reduce computational load
        if len(replay_buffer) > batch_size and step % train_frequency == 0:
            train_start = time.time()
            # Sample a set of random events and actions taken
            batch = random.sample(replay_buffer, batch_size)
            states_batch, actions_batch, rewards_batch, next_states_batch, is_dones_batch = zip(*batch)
            
            # States shape: batch * 4 * 84 * 84
            # Convert tuple of lists to NumPy array and then to tensor
            states_batch = torch.from_numpy(np.array(states_batch, dtype=np.float32)).div(255).to(device)
            # Actions shape: batch * 1
            actions_batch = torch.tensor(actions_batch, dtype=torch.long).unsqueeze(1).to(device)
            # Rewards shape: batch * 1
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
            # Next states shape: batch * 4 * 84 * 84
            # Convert tuple of lists to NumPy array and then to tensor
            next_states_batch = torch.from_numpy(np.array(next_states_batch, dtype=np.float32)).div(255).to(device)
            # Is dones shape: batch * 1
            is_dones_batch = torch.tensor(np.array(is_dones_batch, dtype=np.float32)).to(device)
            
            with torch.autocast(device_type=device):
                # Output of DQN has shape: batch * num_actions
                q_values = dqn(states_batch).gather(1, actions_batch).squeeze()  # Now actions is 2D
            
                with torch.no_grad():
                    # Finds the highest possible next q value for the next state
                    max_next_q_values = target_net(next_states_batch).max(1)[0]
                    target_q_values = rewards_batch + discount_factor * max_next_q_values * (1 - is_dones_batch)
                
                loss = loss_fn(q_values, target_q_values)

            optimiser.zero_grad()
            scaler.scale(loss).backward()

            if step % 20 == 0:
                train_time = time.time() - train_start
                train_times.append(train_time)
                episode_loss.append(loss.item())
                
                # Log to both wandb and files
                log_message(f"Episode: {episode}, Step: {step}, Loss: {loss.item():.6f}, Train time: {train_time:.3f}s")
                with open(log_file, "a") as f:
                    f.write(f"episode: {episode}, step: {step}, loss: {loss.item()}\n")
                
                wandb.log({
                    "loss": loss.item(),
                    "epsilon": epsilon,
                    "step": step,
                    "episode": episode,
                    "train_time": train_time
                })
                    
            scaler.step(optimiser)
            scaler.update()
        
        if step % target_net_update_freq == 0 and step > 0:
            target_net.load_state_dict(dqn.state_dict())
            log_message(f"Target network updated at step {step}")
        
        step += 1
        total_steps += 1
        
        # Print progress every 100 steps
        if total_steps % 100 == 0 and total_steps > 0:
            log_message(f"Total steps: {total_steps}, Epsilon: {epsilon:.4f}")
            
    episode_duration = time.time() - episode_start_time
    episode_rewards.append(episode_reward)
    episode_durations.append(episode_duration)
    
    # Calculate episode metrics
    avg_episode_loss = np.mean(episode_loss) if episode_loss else 0
    episode_losses.append(avg_episode_loss)
    
    # Log episode results
    log_message(f"Episode {episode} finished with reward {episode_reward:.4f} in {episode_duration:.2f} seconds")
    with open(log_file, "a") as f:
        f.write(f"Episode {episode} finished with reward {episode_reward:.4f} in {episode_duration:.2f} seconds\n")
    
    # Log to wandb
    wandb.log({
        "episode_reward": episode_reward,
        "episode_duration": episode_duration,
        "average_episode_loss": avg_episode_loss,
        "total_steps": total_steps,
        "average_reward_last_10": np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards),
        "replay_buffer_size": len(replay_buffer)
    })
    
    # Save model more frequently for early episodes, less frequently for later episodes
    save_interval = 10 if episode < 100 else 100 if episode < 1000 else 500
    if episode % save_interval == 0 and episode > 0:
        # Save models more frequently
        model_path = f"models/{run_name}/dqn_{episode}.pth"
        target_net_path = f"models/{run_name}/target_net_{episode}.pth"
        torch.save(dqn.state_dict(), model_path)
        torch.save(target_net.state_dict(), target_net_path)
        log_message(f"Models saved at episode {episode} to {model_path}")
        
# Save final models
torch.save(dqn.state_dict(), f"models/{run_name}/dqn_latest.pth")
torch.save(target_net.state_dict(), f"models/{run_name}/target_net_latest.pth")
log_message("Training complete. Final models saved.")

# Save training metrics
np.savez(
    f"logs/metrics_{run_name}.npz",
    episode_rewards=np.array(episode_rewards),
    episode_durations=np.array(episode_durations),
    episode_losses=np.array(episode_losses),
    train_times=np.array(train_times)
)

wandb.finish()