"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = irlwpytorch.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import gym
import matplotlib.pyplot as plt
import numpy as np
import logging
import numpy as np
import sys

from MountainCar import MountainCar
from MaxEntropyIRL import MaxEntropyIRL

# from irlwpytorch import __version__

__author__ = "HokageM"
__copyright__ = "HokageM"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

n_states = 400  # position - 20, velocity - 20
n_actions = 3
one_feature = 20  # number of state per one feature
q_table = np.zeros((n_states, n_actions))  # (400, 3)
feature_matrix = np.eye((n_states))  # (400, 400)

gamma = 0.99
q_learning_rate = 0.03
theta_learning_rate = 0.05

np.random.seed(1)


def idx_demo(env, one_feature):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / one_feature

    raw_demo = np.load(file="expert_demo/expert_demo.npy")
    demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))

    for x in range(len(raw_demo)):
        for y in range(len(raw_demo[0])):
            position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
            velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
            state_idx = position_idx + velocity_idx * one_feature

            demonstrations[x][y][0] = state_idx
            demonstrations[x][y][1] = raw_demo[x][y][2]

    return demonstrations


def idx_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / one_feature
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx


def update_q_table(state, action, reward, next_state):
    q_1 = q_table[state][action]
    q_2 = reward + gamma * max(q_table[next_state])
    q_table[state][action] += q_learning_rate * (q_2 - q_1)


q_table = np.load(file="results/maxent_q_table.npy")  # (400, 3)
one_feature = 20  # number of state per one feature


def idx_to_state(env, state):
    """ Convert pos and vel about mounting car environment to the integer value"""
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / one_feature
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * one_feature
    return state_idx


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Implementation of IRL algorithms")
    parser.add_argument(
        "--version",
        action="version",
        # version=f"IRLwPytorch {__version__}",
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    _logger.debug("Starting crazy calculations...")

    car = MountainCar()

    theta = -(np.random.uniform(size=(n_states,)))
    trainer = MaxEntropyIRL(feature_matrix, theta)

    if False:
        env = gym.make('MountainCar-v0', render_mode="human")
        demonstrations = idx_demo(env, one_feature)

        expert = trainer.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(n_states)
        episodes, scores = [], []

        for episode in range(300):
            state = env.reset()
            score = 0

            if (episode != 0 and episode == 100) or (episode > 100 and episode % 50 == 0):
                learner = learner_feature_expectations / episode
                trainer.maxent_irl(expert, learner, theta_learning_rate)

            state = state[0]
            while True:
                state_idx = idx_state(env, state)
                action = np.argmax(q_table[state_idx])
                next_state, reward, done, _, _ = env.step(action)

                irl_reward = trainer.get_reward(n_states, state_idx)
                next_state_idx = idx_state(env, next_state)
                update_q_table(state_idx, action, irl_reward, next_state_idx)

                learner_feature_expectations += trainer.get_feature_matrix()[int(state_idx)]

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 10 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig("./learning_curves/maxent_300.png")
                np.save("./results/maxent_300_table", arr=q_table)

    else:
        env = gym.make('MountainCar-v0', render_mode="human")

        episodes, scores = [], []

        for episode in range(10):
            state = env.reset()
            score = 0

            state = state[0]
            while True:
                env.render()
                state_idx = idx_to_state(env, state)
                action = np.argmax(q_table[state_idx])
                next_state, reward, done, _, _ = env.step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig("./learning_curves/maxent_test_300.png")
                    break

            if episode % 1 == 0:
                print('{} episode score is {:.2f}'.format(episode, score))

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m irlwpytorch.skeleton 42
    #
    run()
