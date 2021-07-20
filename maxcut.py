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
from MAXCUT.maximum_cut import *

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
    default="tf2",
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




class MaximumCut(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config: EnvContext):
        self.seed(50)

        self.num_clusters = config["num_clusters"]
        self.verbose = config["verbose"]
        self.random_clusters_likelihood = config["random_clusters_likelihood"]
        self.done_likelihood = config["done_likelihood"]


        self.mip_model,self.adj,self.nbVar = createOpt()
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
        self.mip_model,self.adj,self.nbVar = createOpt()
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