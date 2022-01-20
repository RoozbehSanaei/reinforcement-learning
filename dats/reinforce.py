import xlrd
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from DATS_Environment import DATS_Environment
from Clustering_Environment import Clustering_Environment

# Cart Pole


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


config = {
    "verbose": True,
    "time_limit": 4,
    "num_clusters": 2,
    "random_clusters_likelihood": 0.2,
    "done_likelihood": 0.001,
    "reduced_dimensionalities": 100,
    "model_name": "DATS",
    "lp_file_name": "LPs/data_12_6_0_r.lp",
    "thresh": 0.5,
    "BATCH_SIZE": 128,
    "GAMMA": 0.999,
    "EPS_START": 0.9,
    "EPS_END": 0.05,
    "EPS_DECAY": 200,
    "TARGET_UPDATE": 10}

device = torch.device("cpu")

'''
env = DATS_Environment(config)
env.seed(args.seed)
torch.manual_seed(args.seed)
'''

book = xlrd.open_workbook('contrastInjector.xls')

sheet_data = book.sheet_by_name('DSM')
data = np.array([[sheet_data.cell_value(r, c) for c in range(
    sheet_data.ncols)] for r in range(sheet_data.nrows)])
DSM = data[1:, 1:].astype(np.float32)
Components = data[1:, 0]

sheet_constraints = book.sheet_by_name('CONSTRAINTS')
constraints_data = np.array([[sheet_constraints.cell_value(r, c) for c in range(
    sheet_constraints.ncols)] for r in range(sheet_constraints.nrows)])
Constraints = constraints_data[1:, 1:].astype(np.float32)


env = Clustering_Environment(DSM, Constraints)
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):

    def __init__(self, h, w, outputs):
        super(Policy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv1.weight.data.to_sparse()
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)
        self.saved_actions = []
        self.rewards = []

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


'''
class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(101, 256)

        # actor's layer
        self.action_head = nn.Linear(256, 4)

        # critic's layer
        self.value_head = nn.Linear(256, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t 
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t 
        return action_prob, state_values
'''
n_actions = (env.N + 1) * (env.N + 1)
model = Policy(env.state.shape[0], env.state.shape[1], n_actions).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()

steps_done = 0


def select_action(state, thresh=0.5):
    global steps_done
    sample = random.random()
    eps_threshold = config["EPS_END"] + (config["EPS_START"] - config["EPS_END"]) * \
        math.exp(-1. * steps_done / config["EPS_DECAY"])
    steps_done += 1
    if sample > 0.5:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action = (model(state)[0] > thresh).int()
            return action
    else:
        return torch.rand(size=(1, n_actions), device=device)


def finish_episode():
    R = 0
    policy_loss = []
    returns = []

    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(model.saved_actions, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            state = torch.Tensor(env.state).to(
                device=device, dtype=torch.float32)[None, None, :, :]
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
