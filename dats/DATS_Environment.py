import copy
import random
from collections import namedtuple
import numpy as np
import matplotlib as mpl
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from DATS import  DATS_instance
from dats.DATS_CPLEX  import DATS_CPLEX
import cplex
import time
from collections import namedtuple
import copy
from sklearn.decomposition import TruncatedSVD

global_counter = 0 

class DATS_Environment():


    def __init__(self, config):
        self.num_clusters = config["num_clusters"]
        self.verbose = config["verbose"]
        self.random_clusters_likelihood = config["random_clusters_likelihood"]
        self.done_likelihood = config["done_likelihood"]
        self.model_name = config["model_name"]
        self.lp_file_name = config["lp_file_name"]
        self.reduced_dimensionalities = config["reduced_dimensionalities"]
        self.clf = TruncatedSVD(n_components=self.reduced_dimensionalities )
        self.history = []



        self.inst = DATS_instance()
#        self.DATS = DATS(self.inst, self.model_name,self.lp_file_name)#
        self.DATS_CPLEX = DATS_CPLEX( self.model_name,self.lp_file_name)
        
        self.mip_model = cplex.Cplex(self.DATS_CPLEX.c)
        self.adj = self.DATS_CPLEX.adjacency.transpose()
        self.var_count = self.DATS_CPLEX.get_nbinvars()


        self.mip_model.max_seconds = config["time_limit"]
        self.mip_model.verbose = config["verbose"]
        self.original_mip_model = cplex.Cplex(self.mip_model)


        #self.var_dict = self.DATS.var_dict_index_name
        self.sol, self.start_obj,status = self.DATS_CPLEX.optimize(init_sol = True)
        self.history.append(self.start_obj)

        Xpca = self.clf.fit_transform(self.adj)

        #self.state = np.concatenate((self.sol[:,np.newaxis],pca.fit_transform(self.adj)),axis=1)
        Xcon = np.concatenate((self.sol[:,np.newaxis],Xpca),axis=1).transpose()


        self.state = self.clf.fit_transform(Xcon)


        
        
        Actions = namedtuple('Actions', 'var_count num_clusters')
        self.actions = Actions(self.var_count,self.num_clusters)
        # Set the seed. This is only used for the final (reach goal) reward.

        self.total_time = 0
        self.start_time = time.time()
        self.cluster_list = []
        self.obj_list = [self.start_obj]

    # def uniform_random_clusters(self,var_dict, num_clusters):
    #     '''Return a random clustering. Each node is assigned to a cluster
    #     a equal probability.'''

    #     choices = list(range(num_clusters))
    #     clusters = dict([(i, []) for i in range(num_clusters)])

    #     for k in var_dict.keys():
    #         cluster_choice = random.choice(choices)
    #         clusters[cluster_choice].append(k)

    #     return clusters

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.mip_model = cplex.Cplex(self.DATS_CPLEX.c) #cplex.Cplex(self.original_mip_model)
        
        self.sol, self.start_obj, status = self.DATS_CPLEX.optimize(True)
        self.history.append(self.start_obj)


        Xpca = self.clf.fit_transform(self.adj)

        #self.state = np.concatenate((self.sol[:,np.newaxis],pca.fit_transform(self.adj)),axis=1)
        Xcon = np.concatenate((self.sol[:,np.newaxis],Xpca),axis=1).transpose()


        self.state = self.clf.fit_transform(Xcon)
        
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

   
        global global_counter
        global_counter = global_counter + 1

        if (random.random()<self.random_clusters_likelihood):
            clusters = self.DATS_CPLEX.uniform_random_clusters(self.num_clusters, True)
        else:
            clusters = self.action_to_clusters(action.cpu().detach().numpy())
            
        #print(global_counter , " == > " , clusters[0])


        self.sol, solver_time, obj, stat = self.DATS_CPLEX.solve_fixed_by_cluster(clusters[0], self.sol )#cplex.Cplex(self.mip_model)
        
        #self.sol, solver_time, obj = self.DATS.solve_fixed_by_cluster(self.mip_model.copy(),clusters[0],self.sol)

        #Sprint(clusters[0],self.sol)
        self.history.append(obj)


        Xpca = self.clf.fit_transform(self.adj)

        #self.state = np.concatenate((self.sol[:,np.newaxis],pca.fit_transform(self.adj)),axis=1)
        Xcon = np.concatenate((self.sol[:,np.newaxis],Xpca),axis=1).transpose()

        self.state = self.clf.fit_transform(Xcon)


        
        self.total_time += solver_time
        if self.verbose:
            print("objective: ", obj)
        self.obj_list.append(obj)
        self.cluster_list.append(clusters)
        reward = -(self.obj_list[-1]-self.obj_list[-2])
        done =  (random.random()<self.done_likelihood)

        return self.state, reward, done, {}
        