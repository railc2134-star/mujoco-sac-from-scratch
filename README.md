# SAC MuJoCo Navigation

A minimal implementation of Soft Actor-Critic (SAC) in PyTorch using MuJoCo.

The agent controls a 2D particle and learns to navigate toward randomly placed targets.

---

## Features

- Soft Actor-Critic (SAC)
- Twin Q-networks
- Target networks with Polyak averaging
- Gaussian policy with tanh squashing
- Replay buffer
- Continuous action space
- MuJoCo visualization

---

## Environment

The environment contains:

- Agent represented as a blue sphere
- Target represented as a purple sphere
- Two slide joints controlling X and Y movement
- Randomized initial positions
- Randomized target positions



## Future Improvements

- Automatic entropy tuning
- Vectorized environments
- Proper Gymnasium wrapper
- TensorBoard logging
- Checkpoint manager
- Support for larger MuJoCo tasks
