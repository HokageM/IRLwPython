import argparse
import logging

import numpy as np
import sys

from irlwpython.MaxEntropyDeep import MaxEntropyDeepIRL
from irlwpython.MountainCar import MountainCar
from irlwpython.MaxEntropyIRL import MaxEntropyIRL

from irlwpython import __version__

__author__ = "HokageM"
__copyright__ = "HokageM"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


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
        version=f"IRLwPython {__version__}",
    )
    parser.add_argument('algorithm', metavar='ALGORITHM', type=str,
                        help='Currently supported training algorithm: [max-entropy, max-entropy-deep]')
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

    n_states = 400  # position - 20, velocity - 20 -> 20*20
    n_actions = 3  # Accelerate to the left: 0, Don’t accelerate: 1, Accelerate to the right: 2
    state_dim = 2  # Velocity and position
    one_feature = 20  # number of state per one feature
    feature_matrix = np.eye(n_states)

    gamma = 0.99
    q_learning_rate = 0.03

    # Theta works as Rewards
    theta_learning_rate = 0.001
    theta = -(np.random.uniform(size=(n_states,)))

    if args.render:
        car = MountainCar(True, one_feature)
    else:
        car = MountainCar(False, one_feature)

    if args.algorithm == "max-entropy-deep" and args.training:
        trainer = MaxEntropyDeepIRL(car, state_dim, n_actions, feature_matrix, one_feature, theta, theta_learning_rate)
        trainer.train(400)

    if args.algorithm == "max-entropy-deep" and args.testing:
        trainer = MaxEntropyDeepIRL(car, 2, n_actions, feature_matrix, one_feature, theta, theta_learning_rate)
        trainer.test("src/irlwpython/results/maxentropydeep_708_best_network_w_-84.0.pth")

    if args.algorithm == "max-entropy" and args.training:
        q_table = np.zeros((n_states, n_actions))
        trainer = MaxEntropyIRL(car, feature_matrix, one_feature, q_table, q_learning_rate, gamma, n_states, theta)
        trainer.train(theta_learning_rate)

    if args.algorithm == "max-entropy" and args.testing:
        q_table = np.load(file="demo/trained_models/qtable_maxentropy_30000_episodes.npy")
        trainer = MaxEntropyIRL(car, feature_matrix, one_feature, q_table, q_learning_rate, gamma, n_states, theta)
        trainer.test()

    _logger.info("Script ends here")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
