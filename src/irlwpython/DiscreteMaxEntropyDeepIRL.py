import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt


class ActorNetwork(nn.Module):
    def __init__(self, num_inputs, num_output, hidden_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_output)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class CriticNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.theta_layer = nn.Linear(hidden_size, 3)

    def forward(self, x):
        x_ = nn.functional.relu(self.fc1(x))
        x_ = nn.functional.relu(self.fc2(x_))
        theta_ = self.theta_layer(x_)
        return self.fc3(x_) + torch.matmul(theta_, x)


class DiscreteMaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_dim, feature_matrix=None, learning_rate=0.01, gamma=0.99,
                 num_epochs=1000):
        self.feat_matrix = feature_matrix
        self.one_feature = 20

        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        self.gamma = gamma
        self.num_epochs = num_epochs
        self.actor_network = ActorNetwork(state_dim, action_dim, 100)
        self.critic_network = CriticNetwork(state_dim + 1, 100)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.Adam(self.critic_network.parameters(), lr=learning_rate)

    def get_reward(self, state, action):
        state_action = list(state) + list([action])
        state_action = torch.Tensor(state_action)
        return self.critic_network(state_action)

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = torch.zeros(400)

        for demonstration in demonstrations:
            for state, _, _ in demonstration:
                state_tensor = torch.tensor(state, dtype=torch.float32)
                feature_expectations += state_tensor.squeeze()

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner):
        # Update critic network

        self.optimizer_critic.zero_grad()

        # Loss function for critic network
        loss_critic = torch.nn.functional.mse_loss(learner, expert) * self.learning_rate
        loss_critic.backward()

        self.optimizer_critic.step()

    def update_q_network(self, state_array, action, reward, next_state):
        self.optimizer_actor.zero_grad()

        state_tensor = torch.tensor(state_array, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

        q_values = self.actor_network(state_tensor)
        q_1 = self.actor_network(state_tensor)[action]

        q_2 = reward + self.gamma * max(self.actor_network(next_state_tensor))
        next_q_values = reward + self.gamma * (q_2 - q_1)  # self.actor_network(next_state_tensor)

        loss_actor = nn.functional.mse_loss(q_values, next_q_values)
        loss_actor.backward()
        self.optimizer_actor.step()

    def train(self):
        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)

        learner_feature_expectations = torch.zeros(400, requires_grad=True)
        episodes, scores = [], []

        for episode in range(self.num_epochs):
            state, info = self.target.env_reset()
            score = 0

            while True:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                q_state = self.actor_network(state_tensor)
                action = torch.argmax(q_state).item()
                next_state, reward, done, _, _ = self.target.env_step(action)

                # Actor update
                irl_reward = self.get_reward(state, action)
                self.update_q_network(state, action, irl_reward, next_state)

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            # Critic update
            state_idx = state[0] + state[1] * self.one_feature
            learner_feature_expectations = learner_feature_expectations + torch.Tensor(
                self.feat_matrix[int(state_idx)])
            learner = learner_feature_expectations / episode
            self.maxent_irl(expert, learner)

            if episode % 1 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig("./learning_curves/discretemaxentdeep_30000.png")

        torch.save(self.actor_network.state_dict(), "./results/discretemaxentdeep_30000_actor.pth")
        torch.save(self.critic_network.state_dict(), "./results/discretemaxentdeep_30000_critic.pth")

    def test(self):
        self.actor_network.load_state_dict(torch.load("./results/discretemaxentdeep_30000_actor.pth"))
        self.critic_network.load_state_dict(torch.load("./results/discretemaxentdeep_30000_critic.pth"))

        episodes, scores = [], []
        for episode in range(10):
            state = self.target.env_reset()
            score = 0

            while True:
                self.target.env_render()
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                action = torch.argmax(self.actor_network(state_tensor)).item()
                next_state, reward, done, _, _ = self.target.env_step(action)

                score += reward
                state = next_state

                if done:
                    scores.append(score)
                    episodes.append(episode)
                    plt.plot(episodes, scores, 'b')
                    plt.savefig("./learning_curves/discretemaxentdeep_test_30000.png")
                    break

            if episode % 1 == 0:
                print('{} episode score is {:.2f}'.format(episode, score))
