# IRLwPython

<img src="logo/IRLwPython.png" width="200">

Inverse Reinforcement Learning Algorithm implementation with python.

# Implemented Algorithms

## Maximum Entropy IRL: [1]

## Maximum Entropy Deep IRL

# Experiments

## Mountaincar-v0
[gym](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)

The expert demonstrations for the Mountaincar-v0 are the same as used in [lets-do-irl](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent).

*Heatmap of Expert demonstrations with 400 states*:

 <img src="demo/heatmaps/expert_state_frequencies_mountaincar.png">

### Maximum Entropy Inverse Reinforcement Learning

IRL using Q-Learning with a Maximum Entropy update function.

#### Training

*Learner training for 29000 episodes*:

<img src="demo/learning_curves/leaner_maxent_29000_episodes.png">

#### Heatmaps

*Learner state frequencies after 1000 episodes*:

<img src="demo/heatmaps/learner_maxent_1000_episodes.png">

*Learner state frequencies after 29000 episodes*:

<img src="demo/heatmaps/leaner_maxent_29000_episodes.png">

*State rewards heatmap after 1000 episodes*:

<img src="demo/heatmaps/rewards_maxent_1000_episodes.png">

*State rewards heatmap after 29000 episodes*:

<img src="demo/heatmaps/rewards_maxent_29000_episodes.png">

#### Testing

*Testing results of the model after 29000 episodes*:

<img src="demo/test_results/test_maxent_29000_episodes.png">


### Deep Maximum Entropy Inverse Reinforcement Learning

IRL using Deep Q-Learning with a Maximum Entropy update function.

#### Training

*Learner training for 29000 episodes*:

<img src="demo/learning_curves/learner_maxentropy_deep_29000_episodes.png">

#### Heatmaps

*Learner state frequencies after 1000 episodes*:

<img src="demo/heatmaps/learner_maxentropydeep_1000_episodes.png">

*Learner state frequencies after 29000 episodes*:

<img src="demo/heatmaps/learner_maxentropydeep_29000_episodes.png">

*State rewards heatmap after 1000 episodes*:

<img src="demo/heatmaps/rewards_maxentropydeep_1000_episodes.png">

*State rewards heatmap after 29000 episodes*:

<img src="demo/heatmaps/rewards_maxentropydeep_29000_episodes.png">

#### Testing

*Testing results of the model after 29000 episodes*:

<img src="demo/test_results/test_maxentropydeep_best_model_results.png">

### Deep Maximum Entropy Inverse Reinforcement Learning with Critic

Coming soon...

# References
The implementation of MaxEntropyIRL and MountainCar is based on the implementation of: 
[lets-do-irl](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent)

[1] [BD. Ziebart, et al., "Maximum Entropy Inverse Reinforcement Learning", AAAI 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf).

# Installation

```commandline
cd IRLwPython
pip install .
```

# Usage

```commandline
usage: irl [-h] [--version] [--training] [--testing] [--render] ALGORITHM

Implementation of IRL algorithms

positional arguments:
  ALGORITHM   Currently supported training algorithm: [max-entropy, max-entropy-deep]

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
  --training  Enables training of model.
  --testing   Enables testing of previously created model.
  --render    Enables visualization of mountaincar.
```
