import json
import os
import pickle
import time
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # Recall: output_size = floor((input_size - kernel_size + 2*padding) / stride) + 1
        # Input shape: batch * 4 * 84 * 84 (batch * frame_stack_size * height * width)
        self.conv1 = nn.Conv2d(4, 32, 8, 4)  # Output shape: batch * 32 * 19 * 19
        # Input shape: batch * 32 * 19 * 19
        self.conv2 = nn.Conv2d(32, 64, 4, 2)  # Output shape: batch * 64 * 9 * 9
        # Input shape: batch * 64 * 9 * 9
        self.conv3 = nn.Conv2d(64, 64, 3, 1)  # Output shape: batch * 64 * 7 * 7
        # Input shape: batch * (64x4x4) (after flattening)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Output shape: batch * 512
        self.fc2 = nn.Linear(512, num_actions)  # Output shape: batch * num_actions
        
    def forward(self, input) -> torch.Tensor:
        # input shape: batch * 4 * 84 * 84
        x = torch.relu(self.conv1(input)) # Output shape: batch * 32 * 19 * 19
        x = torch.relu(self.conv2(x)) # Output shape: batch * 64 * 9 * 9
        x = torch.relu(self.conv3(x)) # Output shape: batch * 64 * 7 * 7
        # Flatten each 3D tensor of the batch into a 1D vector (-1 predicts dim required)
        # batch * 64 * 7 * 7 -> batch * (64x7x7 = 3136)
        x = x.view(x.size(0), -1) # Output shape: batch * (64x7x7 = 3136)
        x = torch.relu(self.fc1(x)) # Output shape: batch * 512
        x = self.fc2(x) # Output shape: batch * num_actions
        return x

# Save and load checkpoint function with replay buffer support
def save_checkpoint(episode, total_steps, run_name, hyperparams, dqn, target_net, optimiser, replay_buffer=None):
    os.makedirs(f"models/{run_name}/checkpoints/ep{episode}", exist_ok=True)
    checkpoint_path = f"models/{run_name}/checkpoints/ep{episode}/checkpoint.pt"
    checkpoint = {
        'episode': episode,
        'total_steps': total_steps,
        'dqn_state_dict': dqn.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimiser_state_dict': optimiser.state_dict(),
        'hyperparams': hyperparams,
        'run_id': run_name,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at episode {episode} to {checkpoint_path}")
    
    # Also create an easy-to-find latest checkpoint
    latest_path = f"models/{run_name}/checkpoints/ep{episode}/checkpoint.pt"
    torch.save(checkpoint, latest_path)
    
    # Save metadata in JSON for easier reading
    metadata = {
        'episode': episode,
        'total_steps': total_steps,
        'learning_rate': optimiser.param_groups[0]['lr'],
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'run_id': run_name,
        'buffer_size': len(replay_buffer) if replay_buffer else 0,
        'has_saved_buffer': replay_buffer is not None
    }
    with open(f"models/{run_name}/checkpoints/ep{episode}/meta.json", 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save replay buffer separately if requested
    # (we don't include this in the main checkpoint to keep file sizes manageable)
    if replay_buffer is not None:
        buffer_path = f"models/{run_name}/checkpoints/ep{episode}/buffer.pkl"
        with open(buffer_path, 'wb') as f:
            pickle.dump(list(replay_buffer), f)
        print(f"Replay buffer saved with {len(replay_buffer)} experiences")