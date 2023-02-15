import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import xlrd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from Clustering_Environment import Clustering_Environment
from torch_geometric.nn import GCNConv
from torch_geometric.nn.aggr import MaxAggregation
from torch.nn.functional import pad

#load DSM and Constraints


book = xlrd.open_workbook('contrastInjector.xls')

sheet_data = book.sheet_by_name('DSM')
data = np.array([[sheet_data.cell_value(r, c) for c in range(sheet_data.ncols)] for r in range(sheet_data.nrows)])
DSM = data[1:,1:].astype(np.float32)
Components = data[1:,0]

sheet_constraints = book.sheet_by_name('CONSTRAINTS')
constraints_data = np.array([[sheet_constraints.cell_value(r, c) for c in range(sheet_constraints.ncols)] for r in range(sheet_constraints.nrows)])
Constraints = constraints_data[1:,1:].astype(np.float32)


g = erdos_renyi_graph(n, 0.2)
DSM = adjacency_matrix(g).todense().astype(np.float32)
c = erdos_renyi_graph(n, 0.01)
Constraints = adjacency_matrix(g).todense().astype(np.float32)
DSM = np.pad(DSM,int((N-n)/2), pad_with,padder=0)
Constraints = np.pad(Constraints, int((N-n)/2), pad_with,padder=0)

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


env =  Clustering_Environment(DSM,Constraints)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = GCNConv(1, 1, cached=True,
                             normalize=True)
        self.linear1 = nn.Linear(1, 1)
        self.linear2 = nn.Linear(1, 1)
        self.affine2 = GCNConv(1, 1, cached=True,
                             normalize=True)
        self.affine3 = GCNConv(1, 1, cached=True,
                             normalize=True)                             

        self.M  = MaxAggregation()

        # actor's layer
        self.action_head = nn.Linear(1, 1)

        # critic's layer
        self.value_head = nn.Linear(1, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x,DSM):
        """
        forward of both actor and critic
        """
        edges_indices = [[],[]]
        edge_values = []
        for edges_start in range(DSM.shape[0]):
            for edges_end in range(DSM.shape[1]):
                edges_indices[0].append(edges_start)
                edges_indices[1].append(edges_end)
                edge_values.append(DSM[edges_start,edges_end])

        t = torch.tensor(edges_indices, dtype=torch.long)
        s = torch.tensor(edge_values, dtype=torch.float)
        x = x[:,:,None]
        x = F.relu(self.affine1(x,edge_index=t,edge_weight=s))
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.affine2(x,edge_index=t))
        x = F.relu(self.affine3(x,edge_index=t))


        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        u = self.M(x,dim_size=1,dim=0)
        v = self.M(u,dim_size=1,dim=-2)


        # critic: evaluates being in the state s_t
        state_values = self.value_head(v)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state,DSM):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state,DSM)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(torch.flatten(probs))

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state  = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 30000):

            # select action from policy
            action = select_action(state,env.DSM)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))

        '''
        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break
        '''


if __name__ == '__main__':
    main()