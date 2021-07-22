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
        #self.grid_width = grid_width
        #self.grid_height = grid_height
        #self.random_start_user_place = random_start_user_place
        #self.random_goal_place = random_goal_place
        #self.blank_space_name = "    "
        #self.wall_space_name = "WALL"
        #self.user_space_name = "USER"
        #self.goal_space_name = "GOAL"
        #self.stochastic_actions_probability = stochastic_actions_probability
        #self.actions = (set(range(self.N)),set(range(Num_Clusters)))
        # Note that the indices of the grid are such that (0, 0) is the top left point
        #self.action_to_effect_dict = {0: "North", 1: "East", 2: "South", 3:"West"}
        #self.current_user_location = None
        #self.current_goal_location = None
        #self.reward_for_achieving_goal = (self.grid_width + self.grid_height) * 3.0
        #self.step_reward_for_not_achieving_goal = -1.0
        #self.state_only_dimension = 1
        #self.num_possible_states = self.grid_height * self.grid_width
        self.action_space = spaces.Discrete((self.N+1)*(self.N+1))
        self.id = "Clustering"
        self.reward_threshold = 0.0
        self.trials = 100


        '''
        if self.random_goal_place:
            self.observation_space = spaces.Dict(dict(
                desired_goal=spaces.Box(0, self.num_possible_states, shape=(1,), dtype='float32'),
                achieved_goal=spaces.Box(0, self.num_possible_states, shape=(1,), dtype='float32'),
                observation=spaces.Box(0, self.num_possible_states, shape=(1,), dtype='float32'),
            ))
        else:
            self.observation_space = spaces.Discrete(self.num_possible_states)

        self.seed()
        self.max_episode_steps = self.reward_for_achieving_goal
        '''


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

        '''
        self.grid = self.create_grid()
        self.place_goal()
        self.place_agent()
        self.desired_goal = [self.location_to_state(self.current_goal_location)]
        self.achieved_goal = [self.location_to_state(self.current_user_location)]
        self.state = [self.location_to_state(self.current_user_location), self.location_to_state(self.current_goal_location)]
        self.achieved_goal = self.state[:self.state_only_dimension]

        if self.random_goal_place:
            self.s = {"observation": np.array(self.state[:self.state_only_dimension]),
                    "desired_goal": np.array(self.desired_goal),
                    "achieved_goal": np.array(self.achieved_goal)}
        else:
            self.s = np.array(self.state)
        '''
        return self.state


    def step(self, desired_action):

        desired_element1 = desired_action // (self.N+1)
        desired_element2 = desired_action % (self.N+1)

        
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
        
        r = self.cost - c


        #print(cost(self.DSM,  seq_2_mat(self.cluster_seq), pow_cc=1))
        if (r>0) and ((self.contraints_violations - contraints_violations) >= 0):    
            self.cost = c
            self.cluster_seq = ns
            self.next_state = np.ndarray.flatten(elementwise_cluster)
            self.contraints_violations = contraints_violations
            self.unchanged = 0
            self.reward = r
        else:
            self.reward = r
            self.next_state = self.state
            self.unchanged = self.unchanged + 1 
        
        self.step_count += 1
        print(f"{self.cost},{self.unchanged}     ",end="\r")
        #("cost: ",self.c
        # ost,"\r",end="")

        
        if ((self.step_count > 10000)): 
            self.done = True
        else:
            self.done = False


        '''
        if type(desired_action) is np.ndarray:
            assert desired_action.shape[0] == 1
            assert len(desired_action.shape) == 1
            desired_action = desired_action[0]
        action = self.determine_which_action_will_actually_occur(desired_action)
        desired_new_state = self.calculate_desired_new_state(action)
        if not self.is_a_wall(desired_new_state):
            self.move_user(self.current_user_location, desired_new_state)
        self.next_state = [self.location_to_state(self.current_user_location), self.desired_goal[0]]

        if self.user_at_goal_location():
            self.reward = self.reward_for_achieving_goal
            self.done = True
        else:
            self.reward = self.step_reward_for_not_achieving_goal
            if self.step_count >= self.max_episode_steps: self.done = True
            else: self.done = False
        self.achieved_goal = self.next_state[:self.state_only_dimension]
        self.state = self.next_state

        if self.random_goal_place:
            self.s = {"observation": np.array(self.next_state[:self.state_only_dimension]),
                "desired_goal": np.array(self.desired_goal),
                "achieved_goal": np.array(self.achieved_goal)}
        else:
            self.s = np.array(self.next_state)
        '''
        self.s = np.array(self.next_state)
        
        
        return self.s, self.reward, self.done, {}

    def determine_which_action_will_actually_occur(self, desired_action):
        """Chooses what action will actually occur. Gives 1. - self.stochastic_actions_probability chance to the
        desired action occuring and the rest of probability spread equally among the other actions"""
        if random.random() < self.stochastic_actions_probability:
            valid_actions = [action for action in self.actions if action != desired_action]
            action = random.choice(valid_actions)
        else: action = desired_action
        return action

    def calculate_desired_new_state(self, action):
        """Calculates the desired new state on basis of action we are going to do"""
        if action == 0:
            desired_new_state = (self.current_user_location[0] - 1, self.current_user_location[1])
        elif action == 1:
            desired_new_state = (self.current_user_location[0], self.current_user_location[1] + 1)
        elif action == 2:
            desired_new_state = (self.current_user_location[0] + 1, self.current_user_location[1])
        elif action == 3:
            desired_new_state = (self.current_user_location[0], self.current_user_location[1] - 1)
        else:
            raise ValueError("Action must be 0, 1, 2, or 3")
        return desired_new_state

    def move_user(self, current_location, new_location):
        """Moves a user from current location to new location"""
        assert self.grid[current_location[0]][current_location[1]] == self.user_space_name, "{} vs. {}".format(self.grid[current_location[0]][current_location[1]], self.user_space_name)
        self.grid[new_location[0]][new_location[1]] = self.user_space_name
        self.grid[current_location[0]][current_location[1]] = self.blank_space_name
        self.current_user_location = (new_location[0], new_location[1])

    def move_goal(self, current_location, new_location):
        """Moves the goal state from current location to new location"""
        assert self.grid[current_location[0]][current_location[1]] == self.goal_space_name
        self.grid[new_location[0]][new_location[1]] = self.goal_space_name
        self.grid[current_location[0]][current_location[1]] = self.blank_space_name
        self.current_goal_location = (new_location[0], new_location[1])

    def is_a_wall(self, location):
        """Returns boolean indicating whether provided location is a wall or not"""
        return self.grid[location[0]][location[1]] == "WALL"

    def return_num_possible_states(self):
        """Returns the number of possible states in this game"""
        return self.grid_width * self.grid_height

    def user_at_goal_location(self):
        """Returns boolean indicating whether user at goal location"""
        return self.current_user_location == self.current_goal_location

    def location_to_state(self, location):
        """Maps a (x, y) location to an integer that uniquely represents its position"""
        return location[0] + location[1] * self.grid_height

    def state_to_location(self, state):
        """Maps a state integer to the (x, y) grid point it represents"""
        col = int(state / self.grid_height)
        row = state - col*self.grid_height
        return (row, col)

    def create_grid(self):
        """Creates and returns the initial gridworld"""
        grid = [[self.blank_space_name for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        centre_col = int(self.grid_width / 2)
        centre_row = int(self.grid_height / 2)
        gaps = [(centre_row, int(centre_col / 2) - 1),  (centre_row, centre_col + int(centre_col / 2)),
                 (int(centre_row/2), centre_col),(centre_row + int(centre_row/2) + 1, centre_col)]
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if row == 0 or col == 0 or row == self.grid_height - 1 or col == self.grid_width - 1:
                    grid[row][col] = self.wall_space_name
                if row == centre_row or col == centre_col:
                    grid[row][col] = self.wall_space_name
                if (row , col) in gaps:
                    grid[row][col] = self.blank_space_name
        return grid

    def place_agent(self):
        """Places the agent on a random non-wall square"""
        if self.random_start_user_place:
            self.current_user_location = self.randomly_place_something(self.user_space_name, [self.wall_space_name, self.goal_space_name])
        else:
            self.current_user_location = (1, 1)
            self.grid[1][1] = self.user_space_name

    def place_goal(self):
        """Places the goal on a random non-WALL and non-USER square"""
        if self.random_goal_place:
            self.current_goal_location = self.randomly_place_something(self.goal_space_name, [self.wall_space_name, self.user_space_name])
        else:
            self.current_goal_location = (3, 3)
            self.grid[3][3] = self.goal_space_name

    def randomly_place_something(self, thing_name, invalid_places):
        """Randomly places a thing called thing_name on any square that doesn't have an invalid item on it"""
        thing_placed = False
        while not thing_placed:
            random_row = randint(0, self.grid_height - 1)
            random_col = randint(0, self.grid_width - 1)
            if self.grid[random_row][random_col] not in invalid_places:
                self.grid[random_row][random_col] = thing_name
                thing_placed = True
        return (random_row, random_col)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward


    def print_current_grid(self):
        """Prints out the grid"""
        for row in range(len(self.grid)):
            print(self.grid[row])

    def visualise_current_grid(self):
        """Visualises the current grid"""
        copied_grid = copy.deepcopy(self.grid)
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if copied_grid[row][col] == self.wall_space_name:
                    copied_grid[row][col] = -100
                elif copied_grid[row][col] == self.blank_space_name:
                    copied_grid[row][col] = 0
                elif copied_grid[row][col] == self.user_space_name:
                    copied_grid[row][col] = 10
                elif copied_grid[row][col] == self.goal_space_name:
                    copied_grid[row][col] = 20
                else:
                    raise ValueError("Invalid values on the grid")
        copied_grid = np.array(copied_grid)
        cmap = mpl.colors.ListedColormap(["black", "white", "blue", "red"])
        bounds = [-101, -1, 1, 11, 21]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        pyplot.imshow(copied_grid, interpolation='nearest',
                            cmap=cmap, norm=norm)
        print("Black = wall, White = empty, Blue = user, Red = goal")
        pyplot.show()
