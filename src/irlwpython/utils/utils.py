# from: https://github.com/reinforcement-learning-kr/lets-do-irl/

import math
import torch
from torch.distributions import Normal


def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    action_list = [0, 1, 2]
    return min(action_list, key=lambda x: abs(x - action))


def get_entropy(mu, std):
    dist = Normal(mu, std)
    entropy = dist.entropy().mean()
    return entropy


def log_prob_density(x, mu, std):
    log_prob_density = -(x - mu).pow(2) / (2 * std.pow(2)) \
                       - 0.5 * math.log(2 * math.pi)
    return log_prob_density.sum(1, keepdim=True)


def get_reward(discrim, state, action):
    print("Input get reward")
    print("state", state)
    print("action", action)

    state = torch.Tensor(state)
    action = torch.Tensor(action)
    state_action = torch.cat([state, action])

    print("HELP")
    print("state", state)
    print("action", action)
    print("state_action", state_action)

    with torch.no_grad():
        return -math.log(discrim(state_action)[0].item())


def save_checkpoint(state, filename):
    torch.save(state, filename)
