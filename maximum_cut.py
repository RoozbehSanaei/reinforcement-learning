"""Example of a custom gym environment and model. Run this for a demo.
This example shows:
  - using a custom environment
  - using a custom model
  - using Tune for grid search
You can visualize experiment results in ~/ray_results using TensorBoard.
"""
import argparse
import gym
from gym.spaces import Discrete, Box,MultiDiscrete
import numpy as np
import os
import random
from ray.rllib.models.tf.misc import normc_initializer
import mip
import networkx as nx
import time
import copy

import ray
from ray import tune
from ray.tune import grid_search
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.")
parser.add_argument(
    "--stop-iters",
    type=int,
    default=50,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=100000,
    help="Number of timesteps to train.")
parser.add_argument(
    "--stop-reward",
    type=float,
    default=0.1,
    help="Reward at which we stop training.")



def gen_graph(max_n, min_n, g_type='barabasi_albert', edge=4):
    cur_n = np.random.randint(max_n - min_n + 1) + min_n
    if g_type == 'erdos_renyi':
        g = nx.erdos_renyi_graph(n = cur_n, p = 0.15)
    elif g_type == 'powerlaw':
        g = nx.powerlaw_cluster_graph(n = cur_n, m = 4, p = 0.05)
    elif g_type == 'barabasi_albert':
        g = nx.barabasi_albert_graph(n = cur_n, m = edge)
    elif g_type == 'watts_strogatz':
        g = nx.watts_strogatz_graph(n = cur_n, k = cur_n // 10, p = 0.1)

    for edge in nx.edges(g):
        g[edge[0]][edge[1]]['weight'] = random.uniform(0,1)

    return g


def getEdgeVar(m, v1, v2, vert):
    u1 = min(v1, v2)
    u2 = max(v1, v2)
    if not ((u1, u2) in vert):
        vert[(u1, u2)] = m.add_var(name='u%d_%d' %(u1, u2),
                                   var_type='B')

    return vert[(u1, u2)]


def getNodeVar(m, v, node):
    if not v in node:
        node[v] = m.add_var(name='v%d' %v,
                            var_type='B')

    return node[v]


def createOpt(G):
    m = mip.Model()
    # Emphasis is on finding good feasible solutions.
    m.emphasis = 1
    edgeVar = {}
    nodeVar = {}
    m.objective = 0
    
 
    
    nbVar = G.number_of_edges()+G.number_of_nodes()
    const_counter = 0
    #create adjacency matrix with one column and add iteratively
    adj = np.ones((nbVar,0))

    


    for j, (v1, v2) in enumerate(G.edges()):
        e12 = getEdgeVar(m, v1, v2, edgeVar)
        node1 = getNodeVar(m, v1, nodeVar)
        node2 = getNodeVar(m, v2, nodeVar)

        m += e12 <= node1 + node2
        
        adj = np.hstack((adj,np.zeros((nbVar,1))))
        adj[v1][const_counter] = -1
        adj[v2][const_counter] = -1
        adj[j+G.number_of_nodes()][const_counter]=1
        const_counter = const_counter + 1
        
        
        m += e12 + node1 + node2 <= 2
    
        adj = np.hstack((adj,np.zeros((nbVar,1))))
        adj[v1][const_counter] = 1
        adj[v2][const_counter] = 1
        adj[j+G.number_of_nodes()][const_counter] = 1
        const_counter = const_counter + 1
    
    
        m.objective = m.objective - (G[v1][v2]['weight']) * e12

        
        
    return m, adj, nbVar







def uniform_random_clusters(var_dict, num_clusters):
    '''Return a random clustering. Each node is assigned to a cluster
    a equal probability.'''

    choices = list(range(num_clusters))
    clusters = dict([(i, []) for i in range(num_clusters)])

    for k in var_dict.keys():
        cluster_choice = random.choice(choices)
        clusters[cluster_choice].append(k)

    return clusters

def generate_var_dict(model):
    '''Returns a dictionary mapping nodes in a graph to variables in a model.'''
    model_vars = model.vars
    num_vars = 0
    var_dict = {}
    

    for model_var in model_vars:
        if model_var.name.startswith('v'):
            num_vars += 1
    var_dict = dict([(i, ["v%d" %i]) for i in range(num_vars)])

    for model_var in model_vars:
        if model_var.name.startswith('u'):
            var_dict[num_vars] = [model_var.name]
            num_vars += 1
    #print(var_dict)
    
    return var_dict


def initialize_solution(var_dict, model):
    '''Initialize a feasible solution.

    Arguments:
      var_dict: a dict maps node index to node variable name.

    Returns:
      a dict maps node variable name to a value.
      a torch tensor view of the dict.
    '''

    sol = {}
    # the sol_vec needs to be of type float for later use with Pytorch.
    sol_vec = None
    init_obj = 0

    for k, var_list in var_dict.items():
        #sol_vec = np.zeros((len(var_dict), len(var_list)))
        for i, v in enumerate(var_list):
            sol[v] = 0      

    return sol, init_obj    






def gradient_descent(model, cluster, var_dict, sol):
    """Perform gradient descent on model along coordinates defined by 
       variables in cluster,  starting from a current solution sol.
    
    Arguments:
      model: the integer program.
      cluster: the coordinates to perform gradient descent on.
      var_dict: mapping node index to node variable name.
      sol: a dict representing the current solution.

    Returns:
      new_sol: the new solution.
      time: the time used to perform the descent.
      obj: the new objective value.
    """

    var_starts = []
    for k, var_list in var_dict.items():
        for v in var_list:
            # warm start variables in the current coordinate set with the existing solution.
            model_var = model.var_by_name(v)
            if k in cluster:
                var_starts.append((model_var, sol[v]))
            else:
                model += model_var == sol[v]

    # model.start = var_starts
    model.verbose = False
    start_time = time.time()
    model.optimize()
    end_time = time.time()
    run_time = end_time - start_time
    new_sol = {}

    for k, var_list in var_dict.items():
        for v in var_list:
            var = model.var_by_name(v)
            try:
                new_sol[v] = round(var.x)
            except:
                return sol, run_time, -1

    return new_sol, run_time, model.objective_value


class MaximumCut(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.seed(50)

        self.num_clusters = config["num_clusters"]
        self.graph = config["graph"]
        self.verbose = config["verbose"]
        self.random_clusters_likelihood = config["random_clusters_likelihood"]
        self.done_likelihood = config["done_likelihood"]


        self.mip_model,self.adj,self.nbVar = createOpt(self.graph)
        self.mip_model.max_seconds = config["time_limit"]
        self.mip_model.verbose = config["verbose"]

        self.var_dict = generate_var_dict(self.mip_model)
        self.sol, self.start_obj = initialize_solution(self.var_dict, self.mip_model)
        self.state = np.concatenate((self.solution_list()[:,np.newaxis],self.adj),axis=1)

        self.action_space = MultiDiscrete(np.ones(self.nbVar)*self.num_clusters)
        self.observation_space =  Box(low=-np.ones(self.state.shape),high=np.ones(self.state.shape))
        # Set the seed. This is only used for the final (reach goal) reward.

        self.total_time = 0
        self.start_time = time.time()
        self.cluster_list = []
        self.obj_list = [self.start_obj]



    def reset(self):
        self.seed(50)
        self.mip_model,self.adj,self.nbVar = createOpt(self.graph)
        self.var_dict = generate_var_dict(self.mip_model)
        self.sol, self.start_obj = initialize_solution(self.var_dict, self.mip_model)
        self.state = np.concatenate((self.solution_list()[:,np.newaxis],self.adj),axis=1)
        return self.state

    def solution_list(self):
        sol_list = np.zeros(len(self.sol))
        for k in self.var_dict: 
            sol_list[k] = self.sol[self.var_dict[k][0]]
        return sol_list

    def action_to_clusters(self,action):
        clusters = dict()
        for k in range(self.num_clusters):
            clusters[k] = [i for i, x in enumerate(action) if x == k]
        return clusters


    def step(self, action):
        #clusters = uniform_random_clusters(self.var_dict,self.num_clusters)
        if (random.random()<self.random_clusters_likelihood):
            clusters = uniform_random_clusters(self.var_dict,self.num_clusters)
        else:
            clusters = self.action_to_clusters(action)

        
        self.sol, solver_time, obj = self.LNS_by_clusters(clusters)
        
        self.state = np.concatenate((self.solution_list()[:,np.newaxis],self.adj),axis=1)
        self.total_time += solver_time
        if self.verbose:
            print("objective: ", obj)
        self.obj_list.append(obj)
        self.cluster_list.append(clusters)
        reward = -(self.obj_list[-1]-self.obj_list[-2])
        done =  (random.random()<self.done_likelihood)



        return self.state, reward, done, {}

    def seed(self, seed=None):
        random.seed(seed)

    def LNS_by_clusters(self,clusters):
        """Perform LNS by clusters. The order of clusters is from the largest to
        the smallest.

        Arguments:
        model: the integer program.
        clusters: a dict mapping cluster index to node index.
        var_dict: mapping node index to node variable name.
        sol: current solution.

        Returns:
        new solution. time spent by the solver. new objective.
        """

        solver_t = 0

        sol, solver_time, obj = gradient_descent(
            self.mip_model.copy(), clusters[0], self.var_dict, self.sol)
        solver_t += solver_time

        return sol, solver_t, obj


class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                name):
        super(CustomModel, self).__init__(obs_space, action_space,
                                        num_outputs, model_config, name)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name="observations")
        layer_1 = tf.keras.layers.Dense(
            128,
            name="my_layer1",
            activation=tf.nn.relu,
            kernel_initializer=normc_initializer(1.0))(self.inputs)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name="my_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(layer_1)
        self.base_model = tf.keras.Model(self.inputs, [layer_out, value_out])

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        if(model_out.shape==(1, 15)):
            model_out =  2*model_out
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

    def metrics(self):
        return {"foo": tf.constant(42.0)}



class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                       model_config, name)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
        fc_out.data[0][0]=0.2
        return fc_out, []

    def value_function(self):
        return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(local_mode=True)

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
        if args.framework == "torch" else CustomModel)

    config = {
        "env": MaximumCut,  # or "corridor" if registered above
        "env_config": {
            "graph": gen_graph(10, 10, g_type='barabasi_albert', edge=6),
            "time_limit": 4,
            "num_clusters": 5,
            "random_clusters_likelihood": 0.2,
            "done_likelihood": 0.001,
            "verbose": True
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "my_model",
            "vf_share_layers": True,
        },
        "lr": 1e-6,  # try different lrs
        "num_workers": 1,  # parallelism
        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()