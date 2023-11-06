import gym
import numpy as np
import matplotlib.pyplot as plt


class MountainCar:

    def __init__(self, animation, feature_matrix, one_feature, q_learning_rate, gamma, n_states, trainer):
        if animation:
            self.env = gym.make('MountainCar-v0', render_mode="human")
        else:
            self.env = gym.make('MountainCar-v0')
        self.feature_matrix = feature_matrix
        self.one_feature = one_feature
        self.q_table = None
        self.q_learning_rate = q_learning_rate
        self.gamma = gamma
        self.n_states = n_states
        self.trainer = trainer

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_q_table(self, table):
        self.q_table = table

    def idx_demo(self, one_feature):
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature

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

    def idx_to_state(self, state):
        """ Convert pos and vel about mounting car environment to the integer value"""
        env_low = self.env.observation_space.low
        env_high = self.env.observation_space.high
        env_distance = (env_high - env_low) / self.one_feature
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
        state_idx = position_idx + velocity_idx * self.one_feature
        return state_idx

    def update_q_table(self, state, action, reward, next_state):
        q_1 = self.q_table[state][action]
        q_2 = reward + self.gamma * max(self.q_table[next_state])
        self.q_table[state][action] += self.q_learning_rate * (q_2 - q_1)

    def env_render(self):
        self.env.render()

    def env_reset(self):
        return self.env.reset()

    def env_step(self, action):
        return self.env.step(action)

    def train(self, theta_learning_rate):
        demonstrations = self.idx_demo(self.one_feature)

        expert = self.trainer.expert_feature_expectations(demonstrations)
        learner_feature_expectations = np.zeros(self.n_states)
        episodes, scores = [], []

        for episode in range(30000):
            state = self.env_reset()
            score = 0

            if (episode != 0 and episode == 10000) or (episode > 10000 and episode % 5000 == 0):
                learner = learner_feature_expectations / episode
                self.trainer.maxent_irl(expert, learner, theta_learning_rate)

            state = state[0]
            while True:
                state_idx = self.idx_to_state(state)
                action = np.argmax(self.q_table[state_idx])
                next_state, reward, done, _, _ = self.env_step(action)

                irl_reward = self.trainer.get_reward(self.n_states, state_idx)
                next_state_idx = self.idx_to_state(next_state)
                self.update_q_table(state_idx, action, irl_reward, next_state_idx)

                learner_feature_expectations += self.trainer.get_feature_matrix()[int(state_idx)]

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
                np.save("./results/maxent_30000_table", arr=self.q_table)

    def test(self):
        episodes, scores = [], []

        for episode in range(10):
            state = self.env_reset()
            score = 0

            state = state[0]
            while True:
                self.env_render()
                state_idx = self.idx_to_state(state)
                action = np.argmax(self.q_table[state_idx])
                next_state, reward, done, _, _ = self.env_step(action)

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
