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
    feature_matrix = np.eye(n_states)  # (400, 400)

    gamma = 0.99
    q_learning_rate = 0.03
    theta_learning_rate = 0.05

    theta = -(np.random.uniform(size=(n_states,)))
    trainer = MaxEntropyIRL(feature_matrix, theta)

    car = None
    if args.render:
        car = MountainCar(True, feature_matrix, one_feature, q_learning_rate, gamma, n_states, trainer)
    else:
        car = MountainCar(False, feature_matrix, one_feature, q_learning_rate, gamma, n_states, trainer)

    if args.training:
        q_table = np.zeros((n_states, n_actions))
        car.set_q_table(q_table)

        car.train(theta_learning_rate)

    if args.testing:
        q_table = np.load(file="./results/maxent_q_table.npy")  # (400, 3)
        car.set_q_table(q_table)

        car.test()

    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
