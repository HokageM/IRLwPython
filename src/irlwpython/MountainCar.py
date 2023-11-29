#
# This file is a refactored implementation of the environment form:
# https://github.com/reinforcement-learning-kr/lets-do-irl/tree/master/mountaincar/maxent
# It is a class type implementation restructured for our use case.
#

import gym
import numpy as np


class MountainCar:

    def __init__(self, animation, one_feature):
        if animation:
            self.env = gym.make('MountainCar-v0', render_mode="human")
        else:
            self.env = gym.make('MountainCar-v0')
        self.one_feature = one_feature

    def get_demonstrations(self):
        """
        Parses the demonstrations and returns the demonstrations.
        :param one_feature:
        :return:
        """
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature

        raw_demo = np.load(file="src/irlwpython/expert_demo/expert_demo.npy")
        demonstrations = np.zeros((len(raw_demo), len(raw_demo[0]), 3))
        for x in range(len(raw_demo)):
            for y in range(len(raw_demo[0])):
                position_idx = int((raw_demo[x][y][0] - env_low[0]) / env_distance[0])
                velocity_idx = int((raw_demo[x][y][1] - env_low[1]) / env_distance[1])
                state_idx = position_idx + velocity_idx * self.one_feature

                demonstrations[x][y][0] = state_idx
                demonstrations[x][y][1] = raw_demo[x][y][2]

        return demonstrations

    def state_to_idx(self, state):
        """
        Converts state (pos, vel) to the integer value using the mountain car environment.
        :param state:
        :return:
        """
        """ """
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_idx = position_idx + velocity_idx * self.one_feature
        return state_idx

    def discretize_state(self, state):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        return [position_idx, velocity_idx]

    def env_action_space(self):
        return self.env.action_space

    def env_observation_space(self):
        return self.env.observation_space

    def env_render(self):
        """
        Computes the render frames as specified by render_mode attribute during initialization of the environment.
        :return:
        """
        self.env.render()

    def env_reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        Start position is in random range of [-0.6, -0.4].
        :return:
        """
        return self.env.reset()

    def env_step(self, action):
        """
        Runs one timestep of the environment's dynamics.
        :param action:
        :return:
        """
        return self.env.step(action)
