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


def fix(seq):
    c = 0
    d = dict()
    n = len(seq)
    for i in range(0,n):
        if not(seq[i] in d.keys()):
            d[seq[i]] = c
            c = c + 1
        seq[i] = d[seq[i]]

def seq_2_mat(seq):
    n = len(seq)
    m = max(seq)+1
    mat = np.zeros((m,n))
    mat[(seq,range(0,n))]=1
    return(mat)

def mat_2_seq(mat):
    seq = [np.nonzero(mat[:,i])[0][0] for i in range(mat.shape[1])]
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
    """Four rooms game environment as described in paper http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf"""
    environment_name = "Four Rooms"

    def __init__(self, DSM,Constraints,Num_Clusters=5):

        self.DSM = DSM
        self.Constraints = Constraints
        self.N = DSM.shape[0]
        self.Num_Clusters = Num_Clusters
        self.cluster_seq = list(range(self.N)) 
        self.cost = cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1) 
        cluster_mat = seq_2_mat(self.cluster_seq)
        cluster_mat_trans = cluster_mat.transpose()
        self.elementwise_cluster = np.dot(cluster_mat_trans,cluster_mat)
        self.contraints_violations = np.sum((self.Constraints)*self.elementwise_cluster) 
        self.state = self.elementwise_cluster
        self.goal = self.cost - 100
        self.step_count = 0
        self.max_steps = 2000000
        self.unchanged = 0
        self.action_space = spaces.Discrete((self.N+1)*(self.N+1))
        self.id = "Clustering"
        self.reward_threshold = 0.0
        self.trials = 100



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""

        self.cluster_seq = list(range(self.N)) 
        self.cost = cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1) 
        cluster_mat = seq_2_mat(self.cluster_seq)
        cluster_mat_trans = cluster_mat.transpose()
        self.elementwise_cluster = np.dot(cluster_mat_trans,cluster_mat)
        self.contraints_violations = np.sum((self.Constraints)*self.elementwise_cluster) 
        self.next_state = None
        self.reward = None
        #if (self.step_count>=20000): self.max_steps = int(self.step_count*1.3)
        self.step_count = 0
        self.done = False
        self.state = self.elementwise_cluster
        self.goal = self.goal - 100
        self.unchanged = 0

        return self.state


    def step(self, action_prob,thresh=0.5):
        n_actions = (self.N + 1) * (self.N + 1)


   
        action_prob_norm = (action_prob-action_prob.min())/(action_prob.max()-action_prob.min())
        desired_action  = (action_prob_norm>thresh).int()
        d = random.choice(np.where(desired_action!=0)[1])

       
        desired_element1 = d // (self.N+1)
        desired_element2 = d % (self.N+1)

        
        ns = self.cluster_seq.copy()
        m = np.max(ns)
        if (desired_element1==self.N):
            if (desired_element2==self.N): 
                self.reset()
            else:   
                ns[desired_element2] = m + 1
        elif (desired_element2==self.N): 
            ns[desired_element1] = m + 1
        else:
            ns[desired_element1] = ns[desired_element2]





        fix(ns)
        cluster_mat = seq_2_mat(ns)
        cluster_mat_trans = cluster_mat.transpose()
        elementwise_cluster = np.dot(cluster_mat_trans,cluster_mat)
        contraints_violations = np.sum((self.Constraints)*elementwise_cluster) 


        c = cost(self.DSM,  seq_2_mat(ns), pow_cc=1)
        
        r = self.cost - c + (self.contraints_violations - contraints_violations)


        #print(cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1))
        if (r>0):    
            self.cost = c
            self.cluster_seq = ns
            self.next_state = elementwise_cluster
            self.contraints_violations = contraints_violations
            self.unchanged = 0
            self.reward = r
        else:
            self.reward = 0
            self.next_state = self.state
            self.unchanged = self.unchanged + 1 
        
        self.step_count += 1
        print(f"{self.cost},{self.unchanged}     ",end="\r")
        #("cost: ",self.c
        # ost,"\r",end="")

        self.s = np.array(self.next_state)
        
        
        return self.s, self.reward, self.done, {}
