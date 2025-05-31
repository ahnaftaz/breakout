# Deep Q-Network (DQN) for Atari Breakout

This project explores the implementation and training of a Deep Q-Network (DQN) agent to play the Atari game Breakout, inspired by DeepMind's work on human-level control through deep reinforcement learning. Below is the trained model demonstrating a tunneling strategy. 

![Breakout Tunneling GIF](/misc/tunneling.gif)

## Overview

The goal was to train an agent capable of playing Breakout effectively using only pixel inputs. The project leverages Q-learning, where an agent learns the value (Quality) of taking actions in different states to maximise cumulative rewards.

As detailed in the [Training a DQN for Breakout](https://ahnaf.bearblog.dev/intro-to-dqn) blog post (coming soon), traditional Q-learning uses tables, which become intractable for high-dimensional state spaces like game screens. Deep Q-Networks (DQNs) address this by using a neural network to approximate the Q-value function, enabling learning directly from raw pixel data.

## Implementation Details

The implementation evolved significantly throughout the training process, incorporating various techniques to improve stability and performance:

- **Core Algorithm:** Started with a standard DQN, later upgraded to **Double DQN (DDQN)** to mitigate Q-value overestimation.
- **Experience Replay:** Utilised a replay buffer to store transitions and sample mini-batches for training, breaking correlations between consecutive experiences. **Prioritised Experience Replay (PER)** was added later.
- **Optimisers & Loss:** Switched from Adam to **AdamW** and from MSE Loss to **Huber Loss** (Smooth L1 Loss) for better stability.
- **Network Architecture:** The CNN architecture processing the game frames was adjusted to optimise feature extraction (details might be in the coming blog post).
- **Environment Handling:** Implemented **vectorised environments** to significantly speed up data collection by running multiple game instances in parallel for a single trainer.
- **Training Stability:** Employed gradient clipping and experimented extensively with hyperparameters like learning rate, batch size, epsilon decay schedule, and buffer size.
- **Reward Shaping:** Experimented with terminating episodes upon losing a life (`terminal mode`) and eventually adopted a multi-stage curriculum training approach to guide learning.

## Training Journey & Challenges

The training process, was an iterative journey filled with challenges (debugging ML is hard):

- **Instability:** Encountered significant training instability, including exploding losses and performance plateaus.
- **Debugging:** Learned the hard way that debugging RL agents requires careful analysis rather than rapid trial-and-error, given the long feedback cycles.
- **Hyperparameter Tuning:** Extensive tuning was required to find a working configuration.
- **Sample Inefficiency:** RL proved to be extremely sample inefficient, requiring millions of steps and many hours of training.
- **Multi-Stage Training:** A final successful strategy involved multiple training stages, adjusting exploration (epsilon), learning rates, and reward structure (terminal mode) to teach the agent different aspects of the game sequentially (general play, surviving life loss, long-term strategy).

## Results

After numerous iterations and approximately 80 hours of cumulative training effort, the agent achieved significant performance on Breakout.

- **Best Checkpoint:** `episode_4750`
- **Average Score (10 runs):** ~209
- **Highest Score Observed:** ~315

It learned the tunneling strategy and successfully employed it in 4/10 games.

![Breakout Tunneling GIF](/misc/tunneling.gif)

While achieving super-human average scores, the agent still exhibited limitations, often mastering specific scenarios but failing in less familiar situations or after achieving high scores. 
- The agent frequently finds itself getting stuck in scenarios it may not have been exposed to.
- Despite the curriculum learning which was intended to teach the agent how to keep playing after losing lives, it still fails to continue playing in 70% of games despite having some lives left.

## Key Takeaways

This project highlighted several key aspects of Reinforcement Learning:

- **Sensitivity:** RL algorithms are highly sensitive to hyperparameters, environment changes, and implementation details.
- **Debugging Difficulty:** Identifying and fixing issues in RL can be challenging due to delayed rewards and complex interactions.
- **Sample Inefficiency:** Training effective agents often requires vast amounts of interaction data.
- **Silent Failures:** RL agents can fail silently, with performance plateauing or degrading without obvious errors as in traditional programming.
- **Importance of Structured Learning:** Curriculum learning or staged training can be crucial for guiding the agent through complex tasks.

This project served as a deep dive into the practical challenges and nuances of training deep reinforcement learning agents.
