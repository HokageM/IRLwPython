import numpy as np
from collections import deque
import torch
import torch.optim as optim
import torch.nn as nn
import os
from utils.utils import *
from utils.zfilter import ZFilter


def train_discrim(discrim, memory, discrim_optim, demonstrations, discrim_update_num):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])

    states = torch.Tensor(states)
    actions = torch.Tensor(actions)

    criterion = torch.nn.BCELoss()

    for _ in range(discrim_update_num):
        learner = discrim(torch.cat([states, actions], dim=1))
        demonstrations = torch.Tensor(demonstrations)
        expert = discrim(demonstrations)

        discrim_loss = criterion(learner, torch.ones((states.shape[0], 1))) + \
                       criterion(expert, torch.zeros((demonstrations.shape[0], 1)))

        discrim_optim.zero_grad()
        discrim_loss.backward()
        discrim_optim.step()

    expert_acc = ((discrim(demonstrations) < 0.5).float()).mean()
    learner_acc = ((discrim(torch.cat([states, actions], dim=1)) > 0.5).float()).mean()

    return expert_acc, learner_acc


def train_actor_critic(actor, critic, memory, actor_optim, critic_optim, actor_critic_update_num, batch_size,
                       clip_param):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    old_values = critic(torch.Tensor(states))
    returns, advants = get_gae(rewards, masks, old_values, args)

    mu, std = actor(torch.Tensor(states))
    old_policy = log_prob_density(torch.Tensor(actions), mu, std)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for _ in range(actor_critic_update_num):
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i: batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)

            inputs = torch.Tensor(states)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -clip_param,
                                         clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()

            loss, ratio, entropy = surrogate_loss(actor, advants_samples, inputs,
                                                  old_policy.detach(), actions_samples,
                                                  batch_index)
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - clip_param,
                                        1.0 + clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()


def get_gae(rewards, masks, values, gamma, lamda):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + (gamma * running_returns * masks[t])
        returns[t] = running_returns

        running_delta = rewards[t] + (gamma * previous_value * masks[t]) - \
                        values.data[t]
        previous_value = values.data[t]

        running_advants = running_delta + (gamma * lamda * \
                                           running_advants * masks[t])
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, batch_index):
    mu, std = actor(states)
    new_policy = log_prob_density(actions, mu, std)
    old_policy = old_policy[batch_index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate_loss = ratio * advants
    entropy = get_entropy(mu, std)

    return surrogate_loss, ratio, entropy


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_outputs)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        logstd = torch.zeros_like(mu)
        std = torch.exp(logstd)
        return mu, std


class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        v = self.fc3(x)
        return v


class Discriminator(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        print("probability", prob)
        return prob


class MaxEntropyDeepIRL:
    def __init__(self, target):
        self.target = target
        torch.manual_seed(523)

    def run(self):
        learning_rate = 0.03
        l2_rate = 0.03
        logdir = 'logs'
        max_iter_num = 4000
        total_sample_size = 2048
        gamma = 0.99
        lamda = 0.98
        clip_param = 0.2
        batch_size = 64
        discrim_update_num = 2
        actor_critic_update_num = 10
        suspend_accu_gen = 0.8
        suspend_accu_exp = 0.8

        # observation space input position and velocity
        num_inputs = self.target.env_observation_space().shape[0]
        # two inputs but 400 different states; position - 20, velocity - 20 -> 20*20 -> num_states

        num_actions = self.target.env_action_space().n
        running_state = ZFilter((num_inputs,), clip=5)

        print('input size:', num_inputs)
        print('action size:', num_actions)

        actor = Actor(num_inputs=num_inputs, num_outputs=1,
                      hidden_size=100)  # 3 different actions but is a scalar as output
        critic = Critic(num_inputs=num_inputs, hidden_size=100)
        discrim = Discriminator(num_inputs=num_inputs, hidden_size=100)

        actor_optim = optim.Adam(actor.parameters(), lr=learning_rate)
        critic_optim = optim.Adam(critic.parameters(), lr=learning_rate, weight_decay=l2_rate)
        discrim_optim = optim.Adam(discrim.parameters(), lr=learning_rate)

        # load demonstrations
        demonstrations = self.target.get_demonstrations()
        # expert_demo, _ = pickle.load(open('./expert_demo/expert_demo.npy.p', "rb"))  # TODO: load same expert demo
        # print(expert_demo)
        # demonstrations = np.array(expert_demo)
        print("demonstrations.shape", demonstrations.shape)

        # writer = SummaryWriter(logdir)

        # TODO: support load model
        # if args.load_model is not None:
        #    saved_ckpt_path = os.path.join(os.getcwd(), 'save_model', str(args.load_model))
        #    ckpt = torch.load(saved_ckpt_path)
        #
        #    actor.load_state_dict(ckpt['actor'])
        #    critic.load_state_dict(ckpt['critic'])
        #    discrim.load_state_dict(ckpt['discrim'])
        #
        #    running_state.rs.n = ckpt['z_filter_n']
        #    running_state.rs.mean = ckpt['z_filter_m']
        #    running_state.rs.sum_square = ckpt['z_filter_s']
        #
        #    print("Loaded OK ex. Zfilter N {}".format(running_state.rs.n))

        episodes = 0
        train_discrim_flag = True

        for iteration in range(max_iter_num):
            actor.eval(), critic.eval()
            memory = deque()

            steps = 0
            scores = []

            while steps < total_sample_size:
                state = self.target.env_reset()
                score = 0

                state = running_state(state[0])

                for _ in range(10000):
                    # if args.render:
                    #    self.target.env_render()

                    steps += 1
                    mu, std = actor(torch.Tensor(state).unsqueeze(0))
                    action = get_action(mu, std)
                    print("Queried action:", action)

                    next_state, reward, done, _, _ = self.target.env_step(action)
                    irl_reward = get_reward(discrim, state, action)

                    if done:
                        mask = 0
                    else:
                        mask = 1

                    memory.append([state, action, irl_reward, mask])

                    next_state = running_state(next_state)
                    state = next_state

                    score += reward

                    if done:
                        break

                episodes += 1
                scores.append(score)

            score_avg = np.mean(scores)
            print('{}:: {} episode score is {:.2f}'.format(iteration, episodes, score_avg))
            # writer.add_scalar('log/score', float(score_avg), iteration)

            actor.train(), critic.train(), discrim.train()
            if train_discrim_flag:
                expert_acc, learner_acc = train_discrim(discrim, memory, discrim_optim, demonstrations,
                                                        discrim_update_num)
                print("Expert: %.2f%% | Learner: %.2f%%" % (expert_acc * 100, learner_acc * 100))
                if expert_acc > suspend_accu_exp and learner_acc > suspend_accu_gen:
                    train_discrim_flag = False
            train_actor_critic(actor, critic, memory, actor_optim, critic_optim, actor_critic_update_num, batch_size,
                               clip_param)

            if iter % 100:
                score_avg = int(score_avg)

                model_path = os.path.join(os.getcwd(), 'save_model')
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)

                ckpt_path = os.path.join(model_path, 'ckpt_' + str(score_avg) + '.pth.tar')

                save_checkpoint({
                    'actor': actor.state_dict(),
                    'critic': critic.state_dict(),
                    'discrim': discrim.state_dict(),
                    'z_filter_n': running_state.rs.n,
                    'z_filter_m': running_state.rs.mean,
                    'z_filter_s': running_state.rs.sum_square,
                    'score': score_avg
                }, filename=ckpt_path)
