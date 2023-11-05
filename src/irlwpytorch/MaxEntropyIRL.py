import numpy as np


class MaxEntropyIRL:
    def __init__(self, feature_matrix, theta):
        self.feature_matrix = feature_matrix
        self.theta = theta

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_feature_matrix(self):
        return self.feature_matrix

    def get_reward(self, n_states, state_idx):
        irl_rewards = self.feature_matrix.dot(self.theta).reshape((n_states,))
        return irl_rewards[state_idx]

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = np.zeros(self.feature_matrix.shape[0])

        for demonstration in demonstrations:
            for state_idx, _, _ in demonstration:
                feature_expectations += self.feature_matrix[int(state_idx)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner, learning_rate):
        gradient = expert - learner
        self.theta += learning_rate * gradient

        # Clip theta
        for j in range(len(self.theta)):
            if self.theta[j] > 0:
                self.theta[j] = 0
