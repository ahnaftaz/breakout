import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import ale_py

from DQN import DQN

# env = gym.make("Breakout-v0", frameskip=1, render_mode="human")
env = gym.make("Breakout-v4", frameskip=1, render_mode="human")
env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=4, noop_max=10)
env = FrameStackObservation(env, stack_size=4)

model = DQN(num_actions=env.action_space.n) # type: ignore
checkpoint = torch.load("models/northern-durian-83/checkpoints/ep4750/checkpoint.pt", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint['dqn_state_dict'])

state, _ = env.reset()

state = torch.tensor(state).float().div(255).unsqueeze(0)

model.eval()

with torch.no_grad():
    action = model(state).argmax(dim=1).item()

terminated = False
action = 0
total_reward = 0.0
state, _, terminated, truncated, _ = env.step(1)
while not terminated:
    next_state, reward, terminated, truncated, info = env.step(action)
    next_state = torch.tensor(next_state).float().div(255).unsqueeze(0)
    with torch.no_grad():
        action = model(next_state)
        print(action)
        action = action.argmax(dim=1).item()

    total_reward += reward.__float__()

print(f"Total reward: {total_reward}")