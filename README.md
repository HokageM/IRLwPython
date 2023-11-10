# IRLwPython


<img src="logo/IRLwPython.jpg" width="200">

Inverse Reinforcement Learning Algorithm implementation with python.

Implemented Algorithms:
- Maximum Entropy IRL
- Maximum Entropy Deep IRL

The implementation of MaxEntropyIRL is based on: https://github.com/reinforcement-learning-kr/lets-do-irl

Mountaincar experiment from: https://www.gymlibrary.dev/environments/classic_control/mountain_car/

# Installation

```commandline
cd IRLwPython
pip install .
```

# Usage

```commandline
usage: irl [-h] [--version] [--training] [--testing] [--render]

Implementation of IRL algorithms

options:
  -h, --help  show this help message and exit
  --version   show program's version number and exit
  --training  Enables training of model.
  --testing   Enables testing of previously created model.
  --render    Enables visualization of mountaincar.

```
