"""
DQN is 
"""
# Standard library imports
import random
from collections import deque
import time
import os

# Third party imports
import ale_py
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import wandb
from DQN import DQN, save_checkpoint
from torchrl.data import PrioritizedReplayBuffer, ListStorage

wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="ahnaft-dev",
    # Set the wandb project where this run will be logged.
    project="breakout",
    # Track hyperparameters and run metadata.
    save_code=True,
)

run_name = wandb.run.name # type: ignore
os.makedirs(f"models/{run_name}", exist_ok=True)
os.makedirs(f"models/{run_name}/loss", exist_ok=True)
os.makedirs(f"models/{run_name}/checkpoints", exist_ok=True)

# Define hyperparameters
hyperparams = {
    "batch_size": 128,  # Increased batch size for more stable updates
    "discount_factor": 0.99,
    "epsilon_min": 0.05,
    "epsilon_start": 1.0,
    "epsilon_decay_steps": 1000000,
    "learning_rate": 2e-4,
    "target_net_update_freq": 10000,
    "buffer_min_size": 131072,  # Minimum buffer size before training starts
    "replay_buffer_size": 1000000,
    "num_episodes": 25000,
    "grad_clip": 10.0,
}

# Log hyperparameters to wandb
wandb.config.update(hyperparams)

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# Initialise the Farama Gym environment
raw_env = gym.make("BreakoutDeterministic-v4", frameskip=1)

# Set up gameplay to skip 4 frames so game feels sped up for model
# Set up screen size as a square to avoid padding requirements
# Set grayscale to reduce channel dimensions with 4 frames to demonstrate velocity
env = AtariPreprocessing(raw_env, frame_skip=4, screen_size=84, grayscale_obs=True)

# Stack the frames so that each observation of the environment state contains 4 frames
# Note that in tandem with the above, the model sees 16 frames for every 1 frame in real time
env = FrameStackObservation(env, stack_size=4)

num_actions = env.action_space.n # type: ignore
        
# Main network that will be optimised at each batch
dqn = DQN(num_actions).to(device)
dqn.train()

wandb.watch(dqn, log="gradients", log_freq=1000)

# A duplicate of an older DQN to stop heavy divergence and improve training stability
target_net = DQN(num_actions).to(device)
target_net.load_state_dict(dqn.state_dict())
target_net.eval()

# Stores the last 1M tuples of (state, action, reward, next_state, done) to sample
# from randomly during training and avoiding overfitting to recent events
# replay_buffer = deque(maxlen=hyperparams["replay_buffer_size"])

replay_buffer = PrioritizedReplayBuffer(
    alpha=0.6, 
    beta=0.4, 
    storage=ListStorage(hyperparams["replay_buffer_size"]),
    batch_size=hyperparams["batch_size"],
)

# Outputs a number between 0-3 (4 possible actions)
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        # Turn into PyTorch tensors and then divide each by 255 to normalise each pixel to
        # between 0-1. Also unsqueeze to add a batch dimension at the start.
        # Input state shape: 4 * 84 * 84
        state_tensor = torch.from_numpy(np.array(state)).float().div(255).unsqueeze(0).to(device)
        # Outut state shape: 1 * 4 * 84 * 84 (with normalised pixels)
        with torch.no_grad():
            q_vals = dqn(state_tensor) # Output shape: 1 * 4
        # Pick the action with the highest quality value
        return q_vals.argmax().item() # Output shape: 1 (single val extracted)

def enter_terminal_mode(env: FrameStackObservation) -> FrameStackObservation:
    raw_env = env.unwrapped
    terminal_env = AtariPreprocessing(raw_env, frame_skip=4, screen_size=84, grayscale_obs=True, terminal_on_life_loss=True)
    return FrameStackObservation(terminal_env, stack_size=4)

num_episodes = hyperparams["num_episodes"]
batch_size = hyperparams["batch_size"]
discount_factor = hyperparams["discount_factor"]
epsilon_min = hyperparams["epsilon_min"]
epsilon_start = hyperparams["epsilon_start"]
target_net_update_freq = hyperparams["target_net_update_freq"]
epsilon_decay_steps = hyperparams["epsilon_decay_steps"]
grad_clip = hyperparams["grad_clip"]
buffer_min_size = hyperparams["buffer_min_size"]

# Train only the DQN and target network will simply copy the state dict occasionally
optimiser = optim.AdamW(dqn.parameters(), lr=hyperparams["learning_rate"], weight_decay=0)
# loss_fn = nn.MSELoss()
# loss_fn = nn.HuberLoss()
loss_fn = nn.SmoothL1Loss()

total_steps = 0
total_eps = 0
ep_reward_history = deque(maxlen=100)
terminal_mode = False

save_checkpoint(total_steps, total_eps, run_name, hyperparams, dqn, target_net, optimiser)
buffer_filling = len(replay_buffer) < hyperparams["buffer_min_size"]
print(f"Buffer filling phase: collecting {hyperparams['buffer_min_size']} experiences before training...")


for episode in range(num_episodes):
    # Outputs state as LazyFrames
    state, info = env.reset()
    # LazyFrames -> np arrays
    state = np.array(state)
    episode_reward = 0
    step = 0
    terminated = False
    episode_start_time = time.time()

    while not terminated:
        epsilon = max(
            epsilon_min, 
            epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay_steps)
        )
        
        action = choose_action(state, epsilon) # Output shape: 1
        next_state, reward, terminated, _, _ = env.step(action)
        next_state = np.array(next_state) # Output shape: 4 * 84 * 84
        reward = np.array(reward) # Output shape: 1
        reward = np.clip(reward, -1, 1)

        # Add a tuple of the current step's state
        replay_buffer.add((state, action, reward, next_state, terminated))
        state = next_state
        episode_reward += reward


        if buffer_filling and len(replay_buffer) >= hyperparams["buffer_min_size"]:
            buffer_filling = False
            print(f"Buffer filled with {len(replay_buffer)} experiences. Starting training...")
        
        if len(replay_buffer) > batch_size and not buffer_filling:
            # Sample a batch sized set of random events and actions taken
            train_start = time.time()
            batch = replay_buffer.sample(batch_size)
            states_batch, actions_batch, rewards_batch, next_states_batch, terminated_batch = zip(*batch)
            
            # States shape: batch * 4 * 84 * 84
            # Convert tuple of lists to NumPy array
            states_batch = torch.from_numpy(np.array(states_batch)).float().div(255).to(device)
            # Actions shape: batch * 1
            actions_batch = torch.tensor(actions_batch).unsqueeze(1).to(device)  # Add an extra dimension
            # Rewards shape: batch * 1
            rewards_batch = torch.tensor(rewards_batch).float().to(device)
            # Next states shape: batch * 4 * 84 * 84
            # Convert tuple of lists to NumPy array
            next_states_batch = torch.from_numpy(np.array(next_states_batch)).float().div(255).to(device)
            # Terminated shape: batch * 1
            terminated_batch = torch.tensor(terminated_batch).float().to(device)
            
            # Output of DQN has shape: batch * num_actions
            q_values_batch = dqn(states_batch).gather(1, actions_batch).squeeze()  # Now actions is 2D

            with torch.no_grad():
                # Finds the highest possible next q value for the next state
                # max_next_q_values_batch = target_net(next_states_batch).max(1)[0]
                # next_target_q_values_batch = target_net(next_states_batch)
                # next_online_q_values_batch = dqn(next_states_batch)
                
                # Outputs from nets in shape (batch, num_acts)
                # Argmax fetches index of highest value in shape (batch,) but keep dim makes it (batch, 1)
                next_action_batch = dqn(next_states_batch).argmax(dim=1, keepdim=True)
                # Gather fetches Q_vals according to argmax index in shape (batch, 1)
                # Squeeze operator removes redundant dimension of size 1 for shape (batch,)
                optimal_q_val = target_net(next_states_batch).gather(1, next_action_batch).squeeze(1)
                # Calculate actual target_q_values (expected future reward of q value)
                target_q_values_batch = rewards_batch + discount_factor * optimal_q_val * (1 - terminated_batch)
                
            loss = loss_fn(q_values_batch, target_q_values_batch)

            optimiser.zero_grad()
            loss.backward()

            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=grad_clip)
            optimiser.step()
            
            train_step_time = time.time() - train_start
            with open(f"models/{run_name}/loss/loss_steps.txt", "a") as f:
                f.write(f"Episode: {episode}, Step: {step}, Loss: {loss.item():.6f}, Train time: {train_step_time:.3f}s\n")

            if step % 50 == 0:        
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epsilon": epsilon,
                        "ep_step": step,
                        "episode": episode,
                        "train_step_time": train_step_time,
                        "total_steps": total_steps,
                        "buffer_size": len(replay_buffer),
                    },
                    step=total_steps
                )

        # Update the target network every 10000 steps
        if total_steps % target_net_update_freq == 0 and not buffer_filling:
            target_net.load_state_dict(dqn.state_dict())

        step += 1
        total_steps += 1
    
    total_eps += 1
    episode_time = time.time() - episode_start_time
    print(f"Episode {episode} finished with reward {episode_reward}, episode time: {episode_time:.3f}s")
    with open(f"models/{run_name}/loss/loss_eps.txt", "a") as f:
        f.write(f"Episode {episode} finished with reward {episode_reward}, episode time: {episode_time:.3f}s\n")
    
    ep_reward_history.append(episode_reward)
    running_avg_ep_reward = np.mean(ep_reward_history)
    wandb.log(
        {
            "episode_time": episode_time,
            "episode_steps": step,
            "episode_reward": episode_reward,
            "episode": episode,
            "total_eps": total_eps,
            "total_steps": total_steps,
            "running_avg_ep_reward": running_avg_ep_reward,
        },
        step=total_steps,
    )

    if episode % 100 == 0 and episode > 0 and not buffer_filling:
        # Don't save the buffer as it's too large. Just waste the money you monkey
        save_buffer = False
        # save_buffer = episode % 2000 == 0
        save_checkpoint(
            episode,
            total_steps, 
            run_name, 
            hyperparams, 
            dqn, 
            target_net, 
            optimiser, 
            replay_buffer if save_buffer else None, 
        )
        
    if episode == 3000 and not terminal_mode:
        env = enter_terminal_mode(env)
        print("ENTERING TERMINAL MODE")
        with open(f"models/{run_name}/loss/loss_eps.txt", "a") as f:
            f.write("ENTERING TERMINAL MODE\n")
        terminal_mode = True

# Final save
save_checkpoint(
    num_episodes-1, 
    total_steps, 
    run_name, 
    hyperparams, 
    dqn, 
    target_net, 
    optimiser, 
    # replay_buffer,
)

os.makedirs(f"models/{run_name}/checkpoints/final", exist_ok=True)
torch.save(dqn.state_dict(), f"models/{run_name}/checkpoints/final/dqn_latest.pth")
torch.save(target_net.state_dict(), f"models/{run_name}/checkpoints/final/target_net_latest.pth")

wandb.finish()
