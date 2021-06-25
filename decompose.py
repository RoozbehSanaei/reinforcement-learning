#!/usr/bin/env python
# coding: utf-8

# # Decomposition for Solving Integer Programs
# ## Introduction
# In this notebook, we will explore the idea of decomposition for solving integer programs (IPs). Integer programming is a general mathematical modeling tools for solving discrete optimization problems. It is widely used in industry, for example, [Amazon](https://www.truckstopsrouting.com/us/amazon-prime-same-day-deliveries/) relies on route optimization for planning deliveries, which can be modelled as IPs. [Sports scheduling](https://www.sciencedirect.com/science/article/abs/pii/S0305054812002869) is another area where IPs are widely used. The bad news is that solving general IPs are [NP-hard](https://en.wikipedia.org/wiki/Integer_programming). So solving an IP to optimality is often prohibitively expensive. In practice, general purpose IP solvers, such as Gurobi and SCIP, are used. These solvers implement the [branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound) tree search algorithm together with a host of heuristics. Typically, these solvers are quite efficient at solving small scale problems with a small number of integer variables. However, once the number of integer variables becomes large, they become inefficient. A natural idea of mitigating this undesirable behavior is through the idea of decomposition. By breaking a large problem into a series of small problems, we can leverage existing strong solvers on the small problems. The main challenge is how to combine solutions generated from the small problems into a feasible solution to the original large problem.

# ## Setup
# We will use the [python-mip](https://www.python-mip.com/) package with an open-source IP solver [CBC](https://projects.coin-or.org/Cbc). We study the problem of computing a [maximum cut (MaxCut)](https://en.wikipedia.org/wiki/Maximum_cut) of a weighted graph. The MaxCut problem is defined over a weighted graph $G = (V, E)$. The problem aims to divide the vertex set $V$ into two disjoint subsets $V = V_0 \cup V_1$ to maximize the total weights of cut edges, edges with one vertex in $V_0$ and the other in $V_1$. This problem can be formulated as IPs
# >$\begin{align}
# & \max \sum_{(i, j)\in E} w_{ij} e_{ij} \\
# & s.t.  \\ 
# & e_{ij} \le v_i + v_j, \forall (i, j) \in E \\
# & e_{ij} + v_i + v_j \le 2, \forall (i, j) \in E \\
# & v_i \in \{0, 1\}, \forall i \in V \\
# & e_{ij} \in \{0, 1\}, \forall (i, j) \in E
# \end{align}
# $
# 
# Intuitively, values for $v_i$ indicate in which subset it is placed. The variables $e_{ij}$ associated with each edge have the value 1 if it is a cut edge and the value 0 otherwise. 
# 
# The following code block generates IPs based on MaxCut problems. It has two components, the **gen_graph** function generates a random graph, with the option to specify which distribution to draw a graph from. Next, the **create_opt** function takes input a graph and outputs an IP based on the formuation above.

# In[5]:


#get_ipython().system('pip install mip')
import mip
import networkx as nx
import numpy as np
import random


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



def generateInstance(max_n, min_n, 
                     g_type='erdos_renyi', edge=4, outPrefix=None):
    G = gen_graph(max_n, min_n, g_type, edge)
    P,adj, nbVar = createOpt(G)

    return G, P, adj, nbVar


# In[6]:


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


def uniform_random_clusters(var_dict, num_clusters):
    '''Return a random clustering. Each node is assigned to a cluster
    a equal probability.'''

    choices = list(range(num_clusters))
    clusters = dict([(i, []) for i in range(num_clusters)])

    for k in var_dict.keys():
        cluster_choice = random.choice(choices)
        clusters[cluster_choice].append(k)

    return clusters


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


# ## Large Neighborhood Search (LNS)
# We now describe the details of our LNS framework. At a high level, our LNS framework operates on an integer program (IP) via defining decompositions of its integer variables into disjoint subsets. Afterwards, we can select a subset and use an existing solver to optimize the variables in that subset while holding all other variables fixed. The benefit of this framework is that it is completely generic to any IP instantiation of any combinatorial optimization problem.
# 
# For an integer program $P$ with a set of integer variables $X$ (not necessarily all the integer variables), we define a decomposition of the set $X$ as a disjoint union $X_1 \cup X_2 \cup \cdots \cup X_k$. Assume we have an existing feasible solution $S_X$ to $P$, we view each subset $X_i$ of integer variables as a local neighborhood for search. We fix integers in $X\setminus X_i$ with their values in the current solution $S_X$ and optimize for variable in $X_i$ (referred as the $\texttt{FIX_AND_OPTIMIZE}$ function in Line 3 of Alg 1). As the resulting optimization is a smaller IP, we can use any off-the-shelf IP solver to carry out the local search. In our experiments, we use Gurobi to optimize the sub-IP. A new solution is obtained and we repeat the process with the remaining subsets.
# 
# <img src="https://drive.google.com/uc?export=view&id=1MJTIMV186ltfohVbZNPArxL9lEueRD4J" height="200"/>

# In[7]:


import time

def LNS(model, num_clusters, steps, time_limit=3, verbose=False):
    """Perform large neighborhood search (LNS) on model. The descent order is random.

    Arguments:
      model: the integer program.
      num_clusters: number of clusters.
      steps: the number of decompositions to apply.
      var_dict: a dict maps node index to node variable name.
      time_limit: time limit for each LNS step.
      sol: (optional) initial solution.
      graph: networkx graph object.
      verbose: if True, print objective after every decomposition.
    """

    model.max_seconds = time_limit
    total_time = 0

    var_dict = generate_var_dict(model)

    start_time = time.time()
    sol, start_obj = initialize_solution(var_dict, model)

    cluster_list = []
    obj_list = [start_obj]

    for _ in range(steps):
        clusters = uniform_random_clusters(var_dict, num_clusters)

        sol, solver_time, obj = LNS_by_clusters(
            model.copy(), clusters, var_dict, sol)
        total_time += solver_time
        if verbose:
            print("objective: ", obj)

        cur_time = time.time()

        cluster_list.append(clusters)
        obj_list.append(obj)

    return total_time, obj, start_obj - obj, cluster_list, obj_list


def LNS_by_clusters(model, clusters, var_dict, sol):
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

    # order clusters by size from largest to smallest
    sorted_clusters = sorted(clusters.items(), 
                             key=lambda x: len(x[1]), 
                             reverse=True)
    solver_t = 0

    for idx, cluster in sorted_clusters:
        sol, solver_time, obj = gradient_descent(
            model.copy(), cluster, var_dict, sol)
        solver_t += solver_time

    return sol, solver_t, obj


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


# Now let's try this idea out with random decompositions. We first generate a random graph sampled according to the [Barabasi-Albert model](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model).

# In[8]:


#G, P, adj, nbVar  = generateInstance(100, 100, g_type='barabasi_albert', edge=5)


# For random decompositions, we randomly decompose the 100 nodes into 5 equally-sized subsets. Iteratively, we apply 10 decompositions in total and for each subproblem, we impose a time limit of 1 second.

# In[9]:


start_time = time.time()
#t, _, _, _, _ = LNS(P, 5, 10, time_limit=1, verbose=True)
end_time = time.time()
#total_t = end_time - start_time
#print('solver time: ', t)
#print('total time: ', total_t)


# To compare the solver's performance without decomposition, we give the solver 10 times the amount of wall-clock time to solve the problem and compare the final objective value.

# In[10]:


#status = P.optimize(max_seconds=total_t * 10)
#print(status)
#print(P.objective_value)


# As the results show, even though the solver uses substantially more time, it produces worse solution compared with the version with decomposition.

# # RL model

# In[11]:


'''
!pip install tensorflow==2.3.0
!pip install gym
!pip install keras
!pip install keras-rl2
'''


# In[12]:


from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


# In[13]:


G, mipmodel, adj, nbVar = generateInstance(50, 50, g_type='barabasi_albert', edge=5)
#mipmodel.optimize()
#print(mipmodel.objective_value)


# In[16]:




from gym import spaces

class ShowerEnv(Env):
    def __init__(self, _nbintvars, _initobservarionstate, _k, _model ):
        
        # get the current state and observation
        self.model = _model
        self.model.verbose = 0
        self.model.optimize(max_solutions = 1)
        self.var_dict = generate_var_dict(_model)
        self.init_sol, self.init_obj = initialize_solution(self.var_dict, _model)
        
        self.init_sol = []
        for k, var_list in self.var_dict.items():
            for v in var_list:
                var = self.model.var_by_name(v)
                self.init_sol.append (round(var.x))
        #print(self.init_sol)
        self.sol = self.init_sol
        self.state = self.init_sol
        

        self.init_obj = self.model.objective_value
        #print(self.init_obj)
        
        
        self.observation =  self.init_obj
        #print(self.init_sol,  self.init_obj)
        print("init obj: ", self.init_obj)
        
        self.action_space = spaces.MultiDiscrete([_k for i in range(nbVar)])
        #self.action_space = spaces.MultiDiscrete((1,_k)*_nbintvars)
        #self.action_space = spaces.Tuple(
        #    tuple(spaces.Discrete(_k) for _ in range(_nbintvars))
        #)
        #self.observation_space = Box(low=np.array([0]), high=np.array([100000]))
        self.observation_space = spaces.MultiDiscrete([2 for i in range(nbVar)])
        #clusters = uniform_random_clusters(var_dict, num_clusters)
        
        # Set episode length
        self.episode_length = 100

    def step(self, action):
        # Apply action
        #print(action)
        #print(self.init_sol)
        
        model = self.model.copy()
        model_vars = self.model.vars
   

        for i  in  range(nbVar):
            if(action[i]>0.5):
                varname = self.var_dict[i][0]
                model_var = model.var_by_name(varname)
                varval = self.sol[i]
                model += model_var == varval
   
       
              
        
        start_time = time.time()
        model.verbose = 0
        model.optimize()
        end_time = time.time()
        run_time = end_time - start_time
        new_sol = []

   
        for k, var_list in self.var_dict.items():
            for v in var_list:
                var = model.var_by_name(v)
                try:
                    new_sol.append ( round(var.x) )
                except:
                    return sol, run_time, -1         
             
        
        self.state =  new_sol
        

        
        # Reduce episode length by 1 second
        self.episode_length -= 1 
        
        # Calculate reward
        
        reward =  - model.objective_value
        #print("reward: ", reward, "\n")
        
        # Check if episode is done
        if self.episode_length <= 0: 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        self.state = new_sol
        
        
        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = self.init_sol #38 + random.randint(-3,3)
        # Reset shower time
        self.episode_length = 100 
        return self.state
    
    #def LNS_by_solver(model, action, var_dict, action):
    #    model


# In[17]:


env = ShowerEnv(_nbintvars = nbVar, _initobservarionstate = 10000, _k = 3, _model = mipmodel)


# In[17]:


#env.action_space.sample()
#del env


# In[18]:


#env.observation_space.sample()


# In[19]:


episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        #print(action)
        n_state, reward, done, info = env.step(action)
        score+=reward
    print('Episode:{} Score:{}'.format(episode, score))


# 2. Create a Deep Learning Model with Keras

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
#actions = env.action_space.n
actions = env.action_space.nvec

def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

model = build_model(states, actions)

model.summary()

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))


_ = dqn.test(env, nb_episodes=15, visualize=True)

# 4. Reloading Agent from Memory

dqn.save_weights('dqn_weights.h5f', overwrite=True)

del model
del dqn
del env

env = gym.make('CartPole-v0')
actions = env.action_space.n
states = env.observation_space.shape[0]
model = build_model(states, actions)
dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('dqn_weights.h5f')
_ = dqn.test(env, nb_episodes=5, visualize=True)