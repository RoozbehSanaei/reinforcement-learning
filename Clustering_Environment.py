import copy
import random
from collections import namedtuple
import gym
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint


def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def fix(seq):
    c = 0
    d = dict()
    n = len(seq)
    for i in range(0, n):
        if not (seq[i] in d.keys()):
            d[seq[i]] = c
            c = c + 1
        seq[i] = d[seq[i]]


def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m, n))
    mat[(seq, range(0, n))] = 1
    return (mat)


def mat_2_seq(mat):
    seq = [np.nonzero(mat[:, i])[0][0] for i in range(mat.shape[1])]
    return (seq)


def cost(DSM,  cluster_matrix, pow_cc=1):
    dsm_size = DSM.shape[0]
    io = np.dot(np.dot(cluster_matrix, DSM), cluster_matrix.transpose())
    ioi = io.diagonal()
    ios = np.sum(io)
    iois = np.sum(ioi)
    ioe = ios - iois
    io_extra = ioe * dsm_size

    cluster_size = np.sum(cluster_matrix, axis=1)
    cscc = np.power(cluster_size, pow_cc)
    io_intra = np.dot(ioi, cscc)
    return (io_extra+io_intra)


class Clustering_Environment(gym.Env):
    environment_name = "Clustering_Environment"

    def __init__(self, DSM, Constraints, Num_Clusters=5, pad_width=0):

        if (pad_width > 0):
            self.DSM = np.pad(DSM, 0, pad_with, padder=0)
            self.Constraints = np.pad(Constraints, 0, pad_with, padder=0)
        else:
            self.DSM = DSM
            self.Constraints = Constraints

        self.training = False
        self.N = self.DSM.shape[0]
        self.Num_Clusters = Num_Clusters
        self.cluster_seq = list(range(self.N))
        self.min_cluster_seq = self.cluster_seq
        self.cost = cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1)
        cluster_mat = seq_2_mat(self.cluster_seq)
        cluster_mat_trans = cluster_mat.transpose()
        self.elementwise_cluster = np.dot(cluster_mat_trans, cluster_mat)
        self.contraints_violations = np.sum(
            (self.Constraints)*self.elementwise_cluster)
        self.state = self.elementwise_cluster
        self.goal = self.cost - 100
        self.step_count = 0
        self.max_steps = 2000000
        self.unchanged = 0
        self.action_space = spaces.Discrete((self.N+1)*(self.N+1))
        self.id = "Clustering"
        self.reward_threshold = 0.0
        self.trials = 100
        self.min_cost = 100000
        self.min_violations = 10000

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""

        self.cluster_seq = list(range(self.N))
        self.min_cluster_seq = self.cluster_seq
        self.cost = cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1)
        cluster_mat = seq_2_mat(self.cluster_seq)
        cluster_mat_trans = cluster_mat.transpose()
        self.elementwise_cluster = np.dot(cluster_mat_trans, cluster_mat)
        self.contraints_violations = np.sum(
            (self.Constraints)*self.elementwise_cluster)
        self.next_state = None
        self.reward = None
        self.step_count = 0
        self.done = False
        self.state = self.elementwise_cluster
        self.goal = self.goal - 100
        self.unchanged = 0
        return self.state

    def step(self, desired_action):

        desired_element1 = desired_action // (self.N+1)
        desired_element2 = desired_action % (self.N+1)

        ns = self.cluster_seq.copy()
        m = np.max(ns)
        if (desired_element1 == self.N):
            if (desired_element2 == self.N):
                self.reset()
            else:
                ns[desired_element2] = m + 1
        elif (desired_element2 == self.N):
            ns[desired_element1] = m + 1
        else:
            ns[desired_element1] = ns[desired_element2]

        fix(ns)
        cluster_mat = seq_2_mat(ns)
        cluster_mat_trans = cluster_mat.transpose()
        elementwise_cluster = np.dot(cluster_mat_trans, cluster_mat)
        contraints_violations = np.sum((self.Constraints)*elementwise_cluster)
        c = cost(self.DSM,  seq_2_mat(ns), pow_cc=1)
        r1 = (self.cost - c)
        r2 = (self.contraints_violations - contraints_violations)

        if ((c < self.min_cost) and (contraints_violations <= self.min_violations)):
            self.min_cost = c
            self.min_violations = contraints_violations
            self.min_cluster_seq = ns

        if ((r1 > 0) or (r2 > 0)):
            self.cost = c
            self.cluster_seq = ns
            self.next_state = elementwise_cluster
            self.contraints_violations = contraints_violations
            self.unchanged = 0
            self.reward = r1 + r2

            if not (self.training):
                print(self.cost, self.contraints_violations,
                      self.min_cost, self.min_violations)
        else:
            self.reward = 0
            self.next_state = self.state
            self.unchanged = self.unchanged + 1

        self.step_count += 1
        self.s = np.array(self.next_state)
        return self.s, self.reward, self.done, {}
