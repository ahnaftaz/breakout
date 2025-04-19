"""
DQN is 
"""
# Standard library imports
import random
from collections import deque
import time
import os
import multiprocessing

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
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

def make_env():
    env = gym.make("BreakoutDeterministic-v4", frameskip=1)
    env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, terminal_on_life_loss=False)
    return FrameStackObservation(env, stack_size=4)

# def make_eval_env():
#     """Creates a Breakout environment for evaluation with terminal_on_life_loss=True."""
#     env = gym.make("BreakoutDeterministic-v4", frameskip=1)
#     # Use terminal_on_life_loss=True for evaluation
#     env = AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True, terminal_on_life_loss=True)
#     return FrameStackObservation(env, stack_size=4)

def choose_vector_actions(states, epsilon, vector_env, dqn, device):
    """Choose actions for multiple environments using vectorized computation."""
    if random.random() < epsilon:
        # Sample random actions for all environments
        return vector_env.action_space.sample()
    else:
        # Convert states to tensor and normalize
        states_tensor = torch.from_numpy(states).float().div(255).to(device)
        with torch.no_grad():
            q_vals = dqn(states_tensor)
        # Return the action with highest Q-value for each environment
        return q_vals.argmax(dim=1).cpu().numpy()

def enter_terminal_mode(num_envs):
    """Switch vector environment to terminal mode (life loss = episode end)"""
    # Create new vector environment with terminal_on_life_loss=True
    return gym.vector.AsyncVectorEnv([
        lambda: gym.wrappers.FrameStackObservation(
            AtariPreprocessing(
                gym.make("BreakoutDeterministic-v4", frameskip=1),
                frame_skip=4, 
                screen_size=84, 
                grayscale_obs=True,
                terminal_on_life_loss=True
            ), 
            stack_size=4
        ) for _ in range(num_envs)  # Use the same number of environments
    ])

# def evaluate_agent(dqn, device, vector_env, num_eval_envs, num_total_eval_episodes=8):
#     """Evaluates the agent's performance over multiple episodes in parallel."""
#     dqn.eval()  # Set the network to evaluation mode

#     total_rewards = []
#     episodes_completed = 0
    
#     # Reset the environments before starting evaluation
#     states, _ = vector_env.reset() 
#     episode_rewards = np.zeros(num_eval_envs)

#     # Use a step limit to prevent infinite loops if episodes don't terminate
#     max_eval_steps = 5000 # Reduced safeguard: ~1 minute of gameplay per environment
#     current_steps = 0

#     while episodes_completed < num_total_eval_episodes and current_steps < max_eval_steps * num_eval_envs:
#         states_tensor = torch.from_numpy(states).float().div(255).to(device)
#         with torch.no_grad():
#             # Use greedy actions during evaluation (no exploration)
#             actions = dqn(states_tensor).argmax(dim=1).cpu().numpy() 
        
#         next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
        
#         episode_rewards += rewards
#         states = next_states
#         current_steps += num_eval_envs

#         # Check for finished episodes in any environment
#         dones = terminated | truncated
#         for i in range(num_eval_envs):
#             if dones[i]:
#                 if episodes_completed < num_total_eval_episodes:
#                     total_rewards.append(episode_rewards[i])
#                     episodes_completed += 1
#                     print(f"Eval episode {episodes_completed}/{num_total_eval_episodes} finished with reward: {episode_rewards[i]}") # Added print
#                 # Reset only the finished environment's reward accumulator
#                 episode_rewards[i] = 0 
#                 # AsyncVectorEnv handles automatic reset internally, but reward needs manual reset
                
#                 # Exit loop early if we have enough episodes
#                 if episodes_completed >= num_total_eval_episodes:
#                     break 

#     dqn.train()  # Set the network back to training mode
    
#     if not total_rewards: # Handle case where no episodes finish 
#         print("Warning: No evaluation episodes finished.")
#         return 0.0
        
#     return np.mean(total_rewards)

def main():
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
        "batch_size": 64,  # Increased batch size for more stable updates
        "discount_factor": 0.99,
        "epsilon_min": 0.05, # Increased minimum exploration
        "epsilon_start": 0.15,
        "epsilon_decay_steps": 500000, # Increased decay steps for more exploration with 8 envs
        "learning_rate": 1e-4,
        "target_net_update_freq": 10000, # Revert to standard/previous value
        "buffer_min_size": 50000,  # Minimum buffer size before training starts
        "replay_buffer_size": 1000000,
        "num_episodes": 50000,
        "grad_clip": 10.0,
        "num_envs": 10,  # Number of parallel environments
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

    # Initialise the Farama Gym vector environment
    vector_env = gym.vector.AsyncVectorEnv([
        lambda: make_env() for _ in range(hyperparams["num_envs"])
    ])

    # # Initialise a separate Farama Gym vector environment for evaluation
    # # Use fewer envs for evaluation to speed it up slightly, and use the eval config
    # num_eval_envs = min(hyperparams["num_envs"], 4) # Use fewer envs for eval
    # eval_vector_env = gym.vector.AsyncVectorEnv([
    #     lambda: make_eval_env() for _ in range(num_eval_envs)
    # ])

    num_actions = vector_env.single_action_space.n # type: ignore
            
    # Main network that will be optimised at each batch
    # dqn = DQN(num_actions).to(device)   # Enable for training from scratch
    checkpoint = torch.load("models/stoic-shadow-82/checkpoints/ep5000/checkpoint.pt", map_location=torch.device('cpu'), weights_only=True)
    dqn = DQN(num_actions).to(device)
    dqn.load_state_dict(checkpoint['dqn_state_dict'])
    dqn.train()

    wandb.watch(dqn, log="gradients", log_freq=1000)

    # A duplicate of an older DQN to stop heavy divergence and improve training stability
    target_net = DQN(num_actions).to(device)
    target_net.load_state_dict(dqn.state_dict())
    target_net.load_state_dict(checkpoint['target_net_state_dict'])
    target_net.eval()


    replay_buffer = PrioritizedReplayBuffer(
        alpha=0.6,
        beta=0.4,
        storage=ListStorage(hyperparams["replay_buffer_size"]),
        batch_size=hyperparams["batch_size"],
        collate_fn=lambda x: x, # Prevent default collation which expects Tensors
    )

    num_episodes = hyperparams["num_episodes"]
    batch_size = hyperparams["batch_size"]
    discount_factor = hyperparams["discount_factor"]
    epsilon_min = hyperparams["epsilon_min"]
    epsilon_start = hyperparams["epsilon_start"]
    target_net_update_freq = hyperparams["target_net_update_freq"]
    epsilon_decay_steps = hyperparams["epsilon_decay_steps"]
    grad_clip = hyperparams["grad_clip"]
    buffer_min_size = hyperparams["buffer_min_size"]
    switch_step_threshold = 1000000

    # Train only the DQN and target network will simply copy the state dict occasionally
    optimiser = optim.AdamW(dqn.parameters(), lr=hyperparams["learning_rate"], weight_decay=0)
    # loss_fn = nn.MSELoss()
    # loss_fn = nn.HuberLoss()
    loss_fn = nn.SmoothL1Loss()

    total_steps = 0
    total_eps = 0
    ep_reward_history = deque(maxlen=100)
    terminal_mode = False
    # eval_freq = 2000 # Evaluate every 2000 steps

    # Restore Beta Annealing Logic
    beta_start = 0.4
    beta_frames = epsilon_decay_steps # Anneal beta over the same period as epsilon
    beta_by_frame = lambda frame_idx: min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)

    save_checkpoint(total_steps, total_eps, run_name, hyperparams, dqn, target_net, optimiser)
    buffer_filling = len(replay_buffer) < buffer_min_size
    print(f"Buffer filling phase: collecting {buffer_min_size} experiences before training...")

    # Reset all environments
    states, _ = vector_env.reset()
    episode_rewards = np.zeros(hyperparams["num_envs"])
    episode_lengths = np.zeros(hyperparams["num_envs"], dtype=int)
    episode_start_times = [time.time()] * hyperparams["num_envs"]
    num_episodes_completed = 0

    while num_episodes_completed < num_episodes:
        epsilon = max(
            epsilon_min, 
            epsilon_start - (epsilon_start - epsilon_min) * (total_steps / epsilon_decay_steps)
        )
        
        # Restore PER beta parameter update
        current_beta = beta_by_frame(total_steps)
        sampler = replay_buffer.sampler
        if isinstance(sampler, PrioritizedSampler):
            sampler.beta = current_beta

        # Choose actions for all environments
        actions = choose_vector_actions(states, epsilon, vector_env, dqn, device)
        
        # Step all environments forward
        next_states, rewards, terminated, truncated, _ = vector_env.step(actions)
        
        # Clip rewards to be between -1 and 1
        rewards = np.clip(rewards, -1, 1)
        
        # Store transitions in replay buffer
        for i in range(hyperparams["num_envs"]):
            replay_buffer.add((
                states[i],
                actions[i],
                rewards[i],
                next_states[i],
                terminated[i] or truncated[i]
            ))
            
            # Update episode rewards and lengths
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            
            # If episode is done, log data and reset
            if terminated[i] or truncated[i]:
                episode_time = time.time() - episode_start_times[i]
                print(f"Episode {num_episodes_completed} finished with reward {episode_rewards[i]}, episode time: {episode_time:.3f}s")
                with open(f"models/{run_name}/loss/loss_eps.txt", "a") as f:
                    f.write(f"Episode {num_episodes_completed} finished with reward {episode_rewards[i]}, episode time: {episode_time:.3f}s\n")
                
                ep_reward_history.append(episode_rewards[i])
                running_avg_ep_reward = np.mean(ep_reward_history)
                wandb.log(
                    {
                        "episode_time": episode_time,
                        "episode_steps": episode_lengths[i],
                        "episode_reward": episode_rewards[i],
                        "episode": num_episodes_completed,
                        "total_eps": total_eps,
                        "total_steps": total_steps,
                        "running_avg_ep_reward": running_avg_ep_reward,
                    },
                    step=total_steps,
                )
                
                # Reset episode tracking for this environment
                episode_rewards[i] = 0
                episode_lengths[i] = 0
                episode_start_times[i] = time.time()
                
                num_episodes_completed += 1
                total_eps += 1
                
                # Switch to terminal mode after switch_step_threshold steps
                if not terminal_mode and total_steps >= switch_step_threshold:
                    print(f"STEP {total_steps}: ENTERING TERMINAL MODE")
                    with open(f"models/{run_name}/loss/loss_eps.txt", "a") as f:
                        f.write(f"STEP {total_steps}: ENTERING TERMINAL MODE\n")
                    # Close old envs first before creating new ones
                    vector_env.close() 
                    # Recreate vector_env with terminal mode enabled
                    vector_env = enter_terminal_mode(hyperparams["num_envs"])
                    # Reset is crucial after recreating the environment
                    states, _ = vector_env.reset() 
                    terminal_mode = True
                    # Reset episode stats as envs have changed
                    episode_rewards = np.zeros(hyperparams["num_envs"])
                    episode_lengths = np.zeros(hyperparams["num_envs"], dtype=int)
                    episode_start_times = [time.time()] * hyperparams["num_envs"]
                
                # Save model checkpoint every 100 episodes completed *after* filling buffer
                if num_episodes_completed % 250 == 0 and num_episodes_completed > 0 and not buffer_filling:
                    save_buffer = False
                    save_checkpoint(
                        num_episodes_completed,
                        total_steps, 
                        run_name, 
                        hyperparams, 
                        dqn, 
                        target_net, 
                        optimiser, 
                        replay_buffer if save_buffer else None,
                    )
                
                # Check if we've reached the desired number of episodes
                if num_episodes_completed >= num_episodes:
                    break
        
        # Update states for next iteration
        states = next_states
        
        # Check if buffer is filled
        if buffer_filling and len(replay_buffer) >= buffer_min_size:
            buffer_filling = False
            print(f"Buffer filled with {len(replay_buffer)} experiences. Starting training...")
        
        # Training step
        if len(replay_buffer) > batch_size and not buffer_filling:
            train_start = time.time()
            # Sample batch - PER returns data and an info dict containing indices
            sampled_data, info_dict = replay_buffer.sample(batch_size, return_info=True)
            states_batch, actions_batch, rewards_batch, next_states_batch, terminated_batch = list(zip(*sampled_data))
            
            # Convert to tensors and normalize states
            # Ensure actions_batch is long type for gather
            states_batch = torch.from_numpy(np.array(states_batch)).float().div(255).to(device)
            actions_batch = torch.tensor(actions_batch, dtype=torch.int64).unsqueeze(1).to(device)
            rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).to(device)
            next_states_batch = torch.from_numpy(np.array(next_states_batch)).float().div(255).to(device)
            terminated_batch = torch.tensor(terminated_batch, dtype=torch.float32).to(device)
            
            # --- Q-Value and Target Calculation (Double DQN) ---
            # Get current Q values for chosen actions
            q_values_batch = dqn(states_batch).gather(1, actions_batch).squeeze()
            
            # Double DQN: use online network to select actions, target network to evaluate
            with torch.no_grad():
                next_action_batch = dqn(next_states_batch).argmax(dim=1, keepdim=True)
                optimal_q_val = target_net(next_states_batch).gather(1, next_action_batch).squeeze(1)
                target_q_values_batch = rewards_batch + discount_factor * optimal_q_val * (1 - terminated_batch)
            
            # --- Loss Calculation and Optimization --- 
            # Compute loss
            loss = loss_fn(q_values_batch, target_q_values_batch)
            
            # --- PER: Update Priorities --- 
            # Calculate TD errors (absolute difference between target and prediction)
            # Detach errors from graph before converting to numpy
            td_errors = (target_q_values_batch - q_values_batch).abs().detach().cpu().numpy()
            # Update priorities in the buffer using the indices from sampling
            replay_buffer.update_priority(info_dict["index"], td_errors)
            
            # --- Backpropagation --- 
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=grad_clip)
            optimiser.step()
            
            train_step_time = time.time() - train_start
            with open(f"models/{run_name}/loss/loss_steps.txt", "a") as f:
                f.write(f"Step: {total_steps}, Loss: {loss.item():.6f}, Train time: {train_step_time:.3f}s\n")
            
            if total_steps % 10 == 0:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "epsilon": epsilon,
                        "beta": current_beta, # Restore beta logging
                        "train_step_time": train_step_time,
                        "total_steps": total_steps,
                        "buffer_size": len(replay_buffer),
                    },
                    step=total_steps
                )
        
        # Update the target network periodically
        if total_steps % target_net_update_freq == 0 and total_steps > 0 and not buffer_filling:
            target_net.load_state_dict(dqn.state_dict())
        
        # # Periodic evaluation
        # if total_steps % eval_freq == 0 and total_steps > 0 and not buffer_filling:
        #     # Evaluate using the dedicated evaluation environment
        #     eval_score = evaluate_agent(dqn, device, eval_vector_env, num_eval_envs) # Pass eval_env and its count
        #     print(f"Step {total_steps}: Evaluation Score (avg per life): {eval_score:.2f}")
        #     wandb.log({"evaluation_score_per_life": eval_score}, step=total_steps)
        
        total_steps += hyperparams["num_envs"]
    
    # Final save
    save_checkpoint(
        num_episodes-1, 
        total_steps, 
        run_name, 
        hyperparams, 
        dqn, 
        target_net, 
        optimiser,
    )

    os.makedirs(f"models/{run_name}/checkpoints/final", exist_ok=True)
    torch.save(dqn.state_dict(), f"models/{run_name}/checkpoints/final/dqn_latest.pth")
    torch.save(target_net.state_dict(), f"models/{run_name}/checkpoints/final/target_net_latest.pth")

    # Close environments
    vector_env.close()
    # eval_vector_env.close()

    wandb.finish()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
