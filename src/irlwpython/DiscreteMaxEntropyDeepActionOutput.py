import math
import os
import random

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import PIL

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(QNetwork, self).__init__()

        # Define the layers of your neural network
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            #nn.ReLU(),  # You can use other activation functions here
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            #nn.Linear(hidden_size, hidden_size),
            #nn.Linear(hidden_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            #nn.Softmax()
        )

    def forward(self, x):
        # Define the forward pass of your neural network
        return self.layers(x)


class DiscreteMaxEntropyDeepIRL:
    def __init__(self, target, state_dim, action_dim, feature_matrix=None, theta=None, learning_rate=0.05, gamma=0.9,
                 num_epochs=1000):
        self.feat_matrix = feature_matrix
        self.one_feature = 20

        self.target = target
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate

        # self.actor_network = ActorNetwork(2, 3, 100)
        self.actor_network = QNetwork(2, 3, 200)
        self.optimizer_actor = optim.Adam(self.actor_network.parameters(), lr=0.01)

        self.gamma = gamma
        self.num_epochs = num_epochs

        self.theta = theta

    def tensor_to_image(self, tensor, name):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        tensor_img = PIL.Image.fromarray(tensor)
        tensor_img.save(f"{name}.png", "PNG")

    def get_reward(self, state):
        state_idx = self.target.state_to_idx(state)
        irl_rewards = self.feat_matrix.dot(self.theta).reshape((400,))
        return irl_rewards[int(state_idx)]

    def expert_feature_expectations(self, demonstrations):
        feature_expectations = np.zeros(self.feat_matrix.shape[0])

        for demonstration in demonstrations:
            for state_idx, _, _ in demonstration:
                feature_expectations += self.feat_matrix[int(state_idx)]

        feature_expectations /= demonstrations.shape[0]
        return feature_expectations

    def maxent_irl(self, expert, learner):
        gradient = expert - learner
        self.theta += self.learning_rate * gradient.detach().numpy()


        #print("EXPERT", expert)
        #print("LEARNER", learner)
        #print("THETA", self.theta)
        self.tensor_to_image(expert.reshape(20,20), "expert_deep")
        self.tensor_to_image(learner.reshape(20,20), "learner_deep")
        self.tensor_to_image(self.theta.reshape(20,20), "theta_deep")

        # Clip theta
        for j in range(len(self.theta)):
            if self.theta[j] > 0:  # log values
                self.theta[j] = 0

    def update_q_network(self, q_state_0: torch.tensor, q_state_1, q_state_2, action, reward, next_q_state):
        q_2 = reward + self.gamma * torch.max(next_q_state)

        if action == 0:
            # q_state_0.requires_grad = True
            q_1 = q_state_0
        #    loss = torch.nn.functional.mse_loss(torch.tensor([q_2, 0, 0], requires_grad=True), q_state)
        if action == 1:
            # q_state_1.requires_grad = True
            q_1 = q_state_1
        #    loss = torch.nn.functional.mse_loss(torch.tensor([0, q_2, 0], requires_grad=True), q_state)
        if action == 2:
            # q_state_2.requires_grad = True
            q_1 = q_state_2
        #    loss = torch.nn.functional.mse_loss(torch.tensor([0, 0, q_2], requires_grad=True), q_state)

        loss = torch.nn.functional.mse_loss(0.01*(q_2 - q_1), q_1)

        self.optimizer_actor.zero_grad()

        loss.backward(retain_graph=True)

        # Print the gradients for each parameter
        for name, param in self.actor_network.named_parameters():
            if name == "layers.3.bias":
                if param.grad is not None:
                    pass
                    #print(name, param.grad)

        self.optimizer_actor.step()

    def train(self):
        demonstrations = self.target.get_demonstrations()
        expert = self.expert_feature_expectations(demonstrations)
        expert = torch.Tensor(expert).detach()
        learner_feature_expectations = torch.zeros(400).detach()
        episodes, scores = [], []
        steps = 0
        for episode in range(20):  # self.num_epochs):
            state, info = self.target.env_reset()
            score = 0

            iteration = 0
            while True:
                steps += 1
                iteration += 1
                if iteration > 2000:
                    #while True:
                    pass
                    #scores.append(-2000)
                    #episodes.append(episode)
                    #break
                state_idx = self.target.state_to_idx(state)
                state_discrete = self.target.discretize_state(state)

                # Forward Pass
                state_tensor = torch.tensor(state_discrete, dtype=torch.float32, requires_grad=True)
                q_state_0, q_state_1, q_state_2 = self.actor_network(state_tensor)

                # Epsilon Greedy Action Selection
                epsilon = 0.1
                if random.uniform(0, 1) > epsilon:
                    #alpha = 0.3
                    #probs = torch.softmax(torch.tensor([q_state_0, q_state_1, q_state_2]) / alpha, dim=0)
                    #action = torch.multinomial(probs, 1).item() # sample action from softmax
                    action = torch.argmax(torch.tensor([q_state_0, q_state_1, q_state_2])).item()
                else:
                    action = random.randint(0, 2)

                next_state, reward, done, _, _ = self.target.env_step(action)
                next_state_discrete = self.target.discretize_state(next_state)
                next_state_tensor = torch.tensor(next_state_discrete, dtype=torch.float32)
                next_q_state = self.actor_network(next_state_tensor).detach()

                # Actor update
                irl_reward = self.get_reward(state)
                #os.system('clear' if os.name == 'posix' else 'cls')
                #print("Episode", episode)
                #print("STATE", state_discrete)
                #print("ACTION", action)
                #print("Q state", q_state_0, q_state_1, q_state_2)
                #print("Next Q state", next_q_state)
                #print("Reward", irl_reward)
                self.update_q_network(q_state_0, q_state_1, q_state_2, action, irl_reward, next_q_state)

                learner_feature_expectations = learner_feature_expectations + torch.Tensor(
                    self.feat_matrix[int(state_idx)])

                score += reward
                state = next_state
                if done:
                    scores.append(score)
                    episodes.append(episode)
                    break

            # Critic update
            if (episode != 0 and episode == 4) or (episode > 4 and episode % 4 == 0):
                learner = learner_feature_expectations / episode
                self.maxent_irl(expert, learner)
            else:
                learner = learner_feature_expectations

            if episode == 19:
                print("Theta", self.theta)
                print("Expert", expert)
                print("Learner", learner)
                for state_1 in range(20):
                    for action_1 in range(20):
                        print("State: ", state_1, action_1, "Q Values",
                              self.actor_network(torch.tensor([state_1, action_1], dtype=torch.float32)),
                              "Reward: ", self.get_reward(state, action))

            if episode % 19 == 0:
                score_avg = np.mean(scores)
                print('{} episode score is {:.2f}'.format(episode, score_avg))
                plt.plot(episodes, scores, 'b')
                plt.savefig("./src/irlwpython/learning_curves/discretemaxentdeep_30000.png")

        torch.save(self.actor_network.state_dict(), "./results/discretemaxentdeep_30000_actor.pth")
