# IRLwPython

<img src="logo/IRLwPython.png" width="200">

Inverse Reinforcement Learning Algorithm implementation with python.

Implemented Algorithms:
- Maximum Entropy IRL: [1]
- Discrete Maximum Entropy Deep IRL: [2, 3]
- IQ-Learn [4]

Experiment:
- Mountaincar: [gym](https://www.gymlibrary.dev/environments/classic_control/mountain_car/)

The implementation of MaxEntropyIRL and MountainCar is based on the implementation of: 
[lets-do-irl](https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent)

# References

[1] [BD. Ziebart, et al., "Maximum Entropy Inverse Reinforcement Learning", AAAI 2008](https://cdn.aaai.org/AAAI/2008/AAAI08-227.pdf).

[2] [Wulfmeier, et al., "Maximum entropy deep inverse reinforcement learning." arXiv preprint arXiv:1507.04888 (2015).](https://arxiv.org/abs/1507.04888)

[3] [Xi-liang Chen, et al., "A Study of Continuous Maximum Entropy Deep Inverse Reinforcement Learning", Mathematical Problems in Engineering, vol. 2019, Article ID 4834516, 8 pages, 2019. https://doi.org/10.1155/2019/4834516](https://www.hindawi.com/journals/mpe/2019/4834516/)

[4] [IQ-learn](https://openreview.net/pdf?id=Aeo-xqtb5p)

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
  ALGORITHM   Currently supported training algorithm: [max-entropy, discrete-max-entropy-deep]

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
  --training  Enables training of model.
  --testing   Enables testing of previously created model.
  --render    Enables visualization of mountaincar.
```
