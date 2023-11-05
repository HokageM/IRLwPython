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
import logging
import numpy as np
import sys

from .MountainCar import MountainCar
from .MaxEntropyIRL import MaxEntropyIRL

# from irlwpytorch import __version__

__author__ = "HokageM"
__copyright__ = "HokageM"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

np.random.seed(1)


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
    parser.add_argument('--training', action='store_true', help="Enables training of model.")
    parser.add_argument('--testing', action='store_true',
                        help="Enables testing of previously created model.")
    parser.add_argument('--render', action='store_true', help="Enables visualization of mountaincar.")
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

    n_states = 400  # position - 20, velocity - 20
    n_actions = 3
    one_feature = 20  # number of state per one feature
    feature_matrix = np.eye((n_states))  # (400, 400)

    gamma = 0.99
    q_learning_rate = 0.03
    theta_learning_rate = 0.05

    car = None
    if args.render:
        car = MountainCar(True, feature_matrix, one_feature, q_learning_rate, gamma)
    else:
        car = MountainCar(False, feature_matrix, one_feature, q_learning_rate, gamma)

    theta = -(np.random.uniform(size=(n_states,)))
    trainer = MaxEntropyIRL(feature_matrix, theta)

    if args.training:
        q_table = np.zeros((n_states, n_actions))  # (400, 3)
        car.set_q_table(q_table)

        demonstrations = car.idx_demo(one_feature)

        expert = trainer.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(n_states)
        episodes, scores = [], []

        for episode in range(30000):
            state = car.env_reset()
            score = 0

            if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
                learner = learner_feature_expectations / episode
                trainer.maxent_irl(expert, learner, theta_learning_rate)

            state = state[0]
            while True:
                state_idx = car.idx_state(state)
                action = np.argmax(q_table[state_idx])
                next_state, reward, done, _, _ = car.env_step(action)

                irl_reward = trainer.get_reward(n_states, state_idx)
                next_state_idx = car.idx_state(next_state)
                car.update_q_table(state_idx, action, irl_reward, next_state_idx)

                learner_feature_expectations += trainer.get_feature_matrix()[int(state_idx)]

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            if episode % 100 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig("./learning_curves/maxent_30000.png")
                np.save("./results/maxent_30000_table", arr=q_table)

    if args.testing:
        q_table = np.load(file="results/maxent_q_table.npy")  # (400, 3)
        car.set_q_table(q_table)

        episodes, scores = [], []

        for episode in range(10):
            state = car.env_reset()
            score = 0

            state = state[0]
            while True:
                car.env_render()
                state_idx = car.idx_to_state(state)
                action = np.argmax(q_table[state_idx])
                next_state, reward, done, _, _ = car.env_step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig("./learning_curves/maxent_test_30000.png")
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
    run()
