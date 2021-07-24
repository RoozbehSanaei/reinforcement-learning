import copy
import random
from collections import namedtuple
import numpy as np
import matplotlib as mpl

from matplotlib import pyplot
from random import randint
from DATS import DATS, DATS_instance
import time
from collections import namedtuple


class DATS_Environment():


    def __init__(self, config):
        self.num_clusters = config["num_clusters"]
        self.verbose = config["verbose"]
        self.random_clusters_likelihood = config["random_clusters_likelihood"]
        self.done_likelihood = config["done_likelihood"]
        self.model_name = config["model_name"]
        self.lp_file_name = config["lp_file_name"]


        self.inst = DATS_instance()
        self.DATS = DATS(self.inst, self.model_name,self.lp_file_name)
        
        self.mip_model,self.adj,self.var_count = self.DATS.m,self.DATS.adjacency,self.DATS.var_count
        self.mip_model.max_seconds = config["time_limit"]
        self.mip_model.verbose = config["verbose"]
        self.original_mip_model = self.mip_model.copy()


        self.var_dict = self.DATS.var_dict_index_name
        self.sol, self.start_obj = self.DATS.optimize()
        self.state = np.concatenate((self.sol[:,np.newaxis],self.adj),axis=1)
        
        Actions = namedtuple('Actions', 'var_count num_clusters')
        self.actions = Actions(self.var_count,self.num_clusters)
        # Set the seed. This is only used for the final (reach goal) reward.

        self.total_time = 0
        self.start_time = time.time()
        self.cluster_list = []
        self.obj_list = [self.start_obj]

    def uniform_random_clusters(self,var_dict, num_clusters):
        '''Return a random clustering. Each node is assigned to a cluster
        a equal probability.'''

        choices = list(range(num_clusters))
        clusters = dict([(i, []) for i in range(num_clusters)])

        for k in var_dict.keys():
            cluster_choice = random.choice(choices)
            clusters[cluster_choice].append(k)

        return clusters

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.mip_model = self.original_mip_model.copy()
        self.sol, self.start_obj = self.DATS.optimize()
        self.state = np.concatenate((self.sol[:,np.newaxis],self.adj),axis=1)
        self.next_state = None
        self.reward = None
        self.done = False

        return self.state

    def action_to_clusters(self,action):
        clusters = dict()
        for k in range(self.num_clusters):
            clusters[k] = [i for i, x in enumerate(action) if x == k]
        return clusters

    def step(self, action):


        if (random.random()<self.random_clusters_likelihood):
            clusters = self.uniform_random_clusters(self.var_dict,self.num_clusters)
        else:
            clusters = self.action_to_clusters(action[0].cpu().detach().numpy())


        self.sol, solver_time, obj = self.DATS.solve_fixed_by_cluster(self.mip_model.copy(), clusters[0], self.sol )
        
        #self.sol, solver_time, obj = self.DATS.solve_fixed_by_cluster(self.mip_model.copy(),clusters[0],self.sol)

        #Sprint(clusters[0],self.sol)

        
        self.state = np.concatenate((self.sol[:,np.newaxis],self.adj),axis=1)
        self.total_time += solver_time
        if self.verbose:
            print("objective: ", obj)
        self.obj_list.append(obj)
        self.cluster_list.append(clusters)
        reward = -(self.obj_list[-1]-self.obj_list[-2])
        done =  (random.random()<self.done_likelihood)

        return self.state, reward, done, {}
        