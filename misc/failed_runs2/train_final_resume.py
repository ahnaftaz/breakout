"""
DQN is 
"""
# Standard library imports
import random
from collections import deque
import time
import os
import argparse
import json
import pickle

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
from DQN import DQN
from tqdm import tqdm

# Add argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
parser.add_argument('--run_id', type=str, help='Wandb run ID to resume')
args = parser.parse_args()

# Initialize wandb with resume capability
if args.resume and args.run_id:
    print(f"Resuming wandb run: {args.run_id}")
    wandb.init(
        entity="ahnaft-dev",
        project="breakout",
        id=args.run_id,
        resume="must",
        save_code=True,
    )
else:
    wandb.init(
        entity="ahnaft-dev",
        project="breakout",
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
    "epsilon_min": 0.1,
    "epsilon_start": 1.0,
    "epsilon_decay_steps": 1000000,
    "learning_rate": 2e-4,
    "target_net_update_freq": 10000,
    "replay_buffer_size": 1000000,
    "buffer_min_size": 200000,  # Minimum buffer size before training starts
    "num_episodes": 20000,
    "grad_clip": 1.0,
    "checkpoint": args.checkpoint if args.resume else None,
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
        
# Main network that will be optimised at each batch
dqn = DQN(num_actions).to(device)
dqn.train()

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
        state_tensor = torch.from_numpy(np.array(state)).float().div(255).unsqueeze(0).to(device)
        # Outut state shape: 1 * 4 * 84 * 84 (with normalised pixels)
        with torch.no_grad():
            q_vals = dqn(state_tensor) # Output shape: 1 * 4
        # Pick the action with the highest quality value
        return q_vals.argmax().item() # Output shape: 1 (single val extracted)

# Setup training variables
num_episodes = hyperparams["num_episodes"]
batch_size = hyperparams["batch_size"]
discount_factor = hyperparams["discount_factor"]
epsilon_min = hyperparams["epsilon_min"]
epsilon_start = hyperparams["epsilon_start"]
target_net_update_freq = hyperparams["target_net_update_freq"]
epsilon_decay_steps = hyperparams["epsilon_decay_steps"]
grad_clip = hyperparams["grad_clip"]

# Train only the DQN and target network will simply copy the state dict occasionally
optimiser = optim.Adam(dqn.parameters(), lr=hyperparams["learning_rate"])
loss_fn = nn.MSELoss()

# Initialize training state
total_steps = 0
start_episode = 0

# Save and load checkpoint function with replay buffer support
def save_checkpoint(episode, total_steps, dqn, target_net, optimiser, replay_buffer=None, save_buffer=False):
    checkpoint_path = f"models/{run_name}/checkpoints/checkpoint_ep{episode}.pt"
    checkpoint = {
        'episode': episode,
        'total_steps': total_steps,
        'dqn_state_dict': dqn.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'hyperparams': hyperparams,
        'run_id': wandb.run.id # type: ignore
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at episode {episode} to {checkpoint_path}")
    
    # Also create an easy-to-find latest checkpoint
    latest_path = f"models/{run_name}/checkpoints/checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Save metadata in JSON for easier reading
    metadata = {
        'episode': episode,
        'total_steps': total_steps,
        'learning_rate': optimiser.param_groups[0]['lr'],
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'run_id': wandb.run.id, # type: ignore
        'buffer_size': len(replay_buffer) if replay_buffer else 0,
        'has_saved_buffer': save_buffer
    }
    with open(f"models/{run_name}/checkpoints/checkpoint_ep{episode}_meta.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save replay buffer separately if requested
    # (we don't include this in the main checkpoint to keep file sizes manageable)
    if save_buffer and replay_buffer:
        buffer_path = f"models/{run_name}/checkpoints/buffer_ep{episode}.pkl"
        with open(buffer_path, 'wb') as f:
            pickle.dump(list(replay_buffer), f)
        print(f"Replay buffer saved with {len(replay_buffer)} experiences")

# Try to load checkpoint if resuming
if args.resume and args.checkpoint:
    print(f"Loading checkpoint from {args.checkpoint}")
    
    # Check if file exists
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint file {args.checkpoint} not found!")
        exit(1)
        
    checkpoint = torch.load(args.checkpoint)
    
    # Load model states
    dqn.load_state_dict(checkpoint['dqn_state_dict'])
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    
    # Load optimizer state
    if 'optimiser_state_dict' in checkpoint and checkpoint['optimiser_state_dict'] is not None:
        optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        print("Optimizer state loaded successfully")
    else:
        print("No optimizer state found in checkpoint, using freshly initialized optimizer")
    
    # Resume from saved training state
    if 'episode' in checkpoint:
        start_episode = checkpoint['episode'] + 1  # Start from next episode
        print(f"Resuming from episode {start_episode}")
    
    if 'total_steps' in checkpoint:
        total_steps = checkpoint['total_steps']
        print(f"Resuming from total steps {total_steps}")
    
    # Set a more moderate learning rate (safer than original 2e-4)
    current_lr = optimiser.param_groups[0]['lr']
    new_lr = 2e-4  # Use a moderate increase from 4.88e-8
    print(f"Adjusting learning rate from {current_lr} to {new_lr}")
    for param_group in optimiser.param_groups:
        param_group['lr'] = new_lr  # Much higher than your current rate
    
    # Try to load replay buffer if it exists
    meta_path = args.checkpoint.replace('.pt', '_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            if meta.get('has_saved_buffer', False):
                buffer_path = args.checkpoint.replace('checkpoint', 'buffer').replace('.pt', '.pkl')
                if os.path.exists(buffer_path):
                    print(f"Loading replay buffer from {buffer_path}")
                    with open(buffer_path, 'rb') as f:
                        buffer_data = pickle.load(f)
                        replay_buffer = deque(buffer_data, maxlen=hyperparams["replay_buffer_size"])
                    print(f"Loaded replay buffer with {len(replay_buffer)} experiences")
                else:
                    print(f"Buffer file {buffer_path} not found, will refill during training")
            else:
                print("No saved buffer found in metadata, will refill during training")

# Save initial checkpoint if not resuming
if not args.resume:
    save_checkpoint(0, 0, dqn, target_net, optimiser)

# Flag to track if we're in buffer filling phase
buffer_filling = len(replay_buffer) < hyperparams["buffer_min_size"]
if buffer_filling:
    print(f"Buffer filling phase: collecting {hyperparams['buffer_min_size']} experiences before training...")

# Main training loop
for episode in range(start_episode, num_episodes):
    # Outputs state as LazyFrames
    state, info = env.reset()
    # LazyFrames -> np arrays
    state = np.array(state)
    episode_reward = 0
    step = 0
    is_done = False
    episode_start_time = time.time()

    while not is_done:
        # Use more exploration during buffer filling
        if buffer_filling:
            # Higher epsilon during buffer filling (linear decay from 1.0 to 0.1)
            fill_progress = min(1.0, len(replay_buffer) / hyperparams["buffer_min_size"]) 
            epsilon = max(
                epsilon_min,
                epsilon_start - (epsilon_start - epsilon_min) * fill_progress
            )
        else:
            epsilon = max(
                epsilon_min, 
                epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay_steps)
            )
        
        action = choose_action(state, epsilon) # Output shape: 1
        next_state, reward, is_done, _, _ = env.step(action)
        next_state = np.array(next_state) # Output shape: 4 * 84 * 84
        reward = np.array(reward) # Output shape: 1
        reward = np.clip(reward, -1, 1)

        # Add a tuple of the current step's state
        replay_buffer.append((state, action, reward, next_state, is_done))
        state = next_state
        episode_reward += reward
        
        # Check if buffer is now filled 
        if buffer_filling and len(replay_buffer) >= hyperparams["buffer_min_size"]:
            buffer_filling = False
            print(f"Buffer filled with {len(replay_buffer)} experiences. Starting training...")
            # Save the filled buffer
            save_checkpoint(episode, total_steps, dqn, target_net, optimiser, replay_buffer, save_buffer=True)
        
        # Only train if buffer is sufficiently filled
        if not buffer_filling and len(replay_buffer) > batch_size:
            # Sample a batch sized set of random events and actions taken
            train_start = time.time()
            batch = random.sample(replay_buffer, batch_size)
            states_batch, actions_batch, rewards_batch, next_states_batch, is_dones_batch = zip(*batch)
            
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
            # Is dones shape: batch * 1
            is_dones_batch = torch.tensor(is_dones_batch).float().to(device)
            
            # Output of DQN has shape: batch * num_actions
            q_values_batch = dqn(states_batch).gather(1, actions_batch).squeeze()  # Now actions is 2D
            
            with torch.no_grad():
                # Finds the highest possible next q value for the next state
                max_next_q_values_batch = target_net(next_states_batch).max(1)[0]
                target_q_values_batch = rewards_batch + discount_factor * max_next_q_values_batch * (1 - is_dones_batch)
                
            loss = loss_fn(q_values_batch, target_q_values_batch)

            optimiser.zero_grad()
            loss.backward()
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=grad_clip)
            optimiser.step()
            
            train_step_time = time.time() - train_start
            with open(f"models/{run_name}/loss/loss_steps.txt", "a") as f:
                f.write(f"Episode: {episode}, Step: {step}, Loss: {loss.item():.6f}, Train time: {train_step_time:.3f}s\n")
            
            # Log current learning rate
            if step % 1000 == 0:
                current_lr = optimiser.param_groups[0]['lr']
                wandb.log({
                    "learning_rate": current_lr,
                    "total_steps": total_steps,
                    "episode": episode
                })
            
            wandb.log({
                "loss": loss.item(),
                "epsilon": epsilon,
                "step": step,
                "episode": episode,
                "train_step_time": train_step_time,
                "total_steps": total_steps,
                "buffer_size": len(replay_buffer),
            })

        
        if total_steps % target_net_update_freq == 0 and not buffer_filling:
            target_net.load_state_dict(dqn.state_dict())
            
        step += 1
        total_steps += 1
        
        if is_done:
            break
            
    episode_time = time.time() - episode_start_time
    print(f"Episode {episode} finished with reward {episode_reward}, episode time: {episode_time:.3f}s")
    with open(f"models/{run_name}/loss/loss_eps.txt", "a") as f:
        f.write(f"Episode {episode} finished with reward {episode_reward}, episode time: {episode_time:.3f}s\n")
    
    wandb.log({
        "episode_time": episode_time,
        "episode_steps": step,
        "episode_reward": episode_reward,
        "episode": episode,
        "buffer_size": len(replay_buffer),
    })
    
    # Save comprehensive checkpoint every 100 episodes
    if episode % 100 == 0 and episode > 0 and not buffer_filling:
        # Save replay buffer every 500 episodes to avoid too many large files
        save_buffer = (episode % 500 == 0)
        save_checkpoint(episode, total_steps, dqn, target_net, optimiser, replay_buffer, save_buffer=save_buffer)

# Final save
save_checkpoint(num_episodes-1, total_steps, dqn, target_net, optimiser, replay_buffer, save_buffer=True)
torch.save(dqn.state_dict(), f"models/{run_name}/dqn_latest.pth")
torch.save(target_net.state_dict(), f"models/{run_name}/target_net_latest.pth")

wandb.finish()
