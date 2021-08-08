# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2018
# --------------------------------------------------------------------------

# The goal of the diet problem is to select a set of foods that satisfies
# a set of daily nutritional requirements at minimal cost.
# Source of data: http://www.neos-guide.org/content/diet-problem-solver

from collections import namedtuple

from docplex.mp.model import Model
from docplex.util.environment import get_environment
import sys
import cplex
import numpy as np
import random
import copy
import time
import numpy as np
from scipy.sparse import coo_matrix


class DATS_CPLEX:
    class StopCriterion(cplex.callbacks.MIPInfoCallback):
        def __call__(self):
            if self.has_incumbent(): ## self.get_num_nodes() > 1000:
                #if self.get_MIP_relative_gap() < 0.1:
                self.abort()
                return
            #else: # we’ve processed fewer than 1000 nodes
            #    if self.get_MIP_relative_gap() < 0.001:
            #        self.abort()
                #return

    def __init__(self,model_name, lp_file_name):
        # Read in a model file.
        self.c = cplex.Cplex()
        self.c.register_callback(self.StopCriterion)
        self.c.read(lp_file_name)

        ## deteriorate CPLEX performance, disable later
        #self.c.context.cplex_parameters.threads = 1
        #self.c.context.cplex_parameters.threads = 1
        self.c.parameters.mip.strategy.heuristicfreq = -1
        self.c.parameters.parallel.set(-1)
        self.c.parameters.mip.cuts.bqp.set(-1)
        self.c.parameters.mip.cuts.cliques.set(-1)
        self.c.parameters.mip.cuts.covers.set(-1)
        self.c.parameters.mip.cuts.disjunctive.set(-1)
        self.c.parameters.mip.cuts.flowcovers.set(-1)
        self.c.parameters.mip.cuts.pathcut.set(-1)
        self.c.parameters.mip.cuts.gomory.set(-1)
        self.c.parameters.mip.cuts.gubcovers.set(-1)
        self.c.parameters.mip.cuts.implied.set(-1)
        self.c.parameters.mip.cuts.localimplied.set(-1)
        self.c.parameters.mip.cuts.liftproj.set(-1)
        self.c.parameters.mip.cuts.mircut.set(-1)
        self.c.parameters.mip.cuts.mcfcut.set(-1)
        self.c.parameters.mip.cuts.rlt.set(-1)
        self.c.parameters.mip.cuts.zerohalfcut.set(-1)

        self.c.parameters.timelimit.set(100)


        ##-------------------------------------------

        # # Display all binary variables that have a solution value of 1.
        types = self.c.variables.get_types()
        self.nvars = self.c.variables.get_num()
        nconsts = self.c.linear_constraints.get_num()
        rows = self.c.linear_constraints.get_rows()

        #var_types = self.c.variables.get_types()

        self.binvars = [idx for idx, typ
            in zip(range(self.nvars), self.c.variables.get_types())
            if typ == self.c.variables.type.binary]
        #get binary variables' names
        self.binvarnames = self.c.variables.get_names(self.binvars)

        # get columns of binary vars
        binvarcols = self.c.variables.get_cols(self.binvars)


        sparse_rows = []
        sparse_cols = []
        sparse_val = []
        for v, e in zip(range(self.get_nbinvars()), binvarcols):
            for ind in e.ind:
                sparse_rows.append(v)
                sparse_cols.append(ind)
            for val in e.val:
                sparse_val.append(val)
        

        self.adjacency = coo_matrix((sparse_val, (sparse_rows, sparse_cols)))


        return None

    def get_nbinvars(self):
        return len(self.binvars)
	
    def uniform_random_clusters(self,  num_clusters):
        '''Return a random clustering. Each node is assigned to a cluster
        a equal probability.'''

        choices = list(range(num_clusters))
        clusters = dict([(i, []) for i in range(num_clusters)])

        for k in self.binvars:
            cluster_choice = random.choice(choices)
            clusters[cluster_choice].append(k)

        return clusters


    def solve_fixed_by_cluster(self, model, cluster, sol=None):
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


        # var_starts = []
        # for i in self.binvars:
        #     if i in cluster:
        #         var_starts.append((i, sol[i]))
        #     else:
        #         self.c.variables.set_lower_bounds(i, sol[i])
        #         self.c.variables.set_upper_bounds(i, sol[i])

        #for var, val in zip (cluster, sol[cluster]):   
        self.c.variables.set_upper_bounds([(var, 1.0) for var  in self.binvars])
        self.c.variables.set_lower_bounds([(var, 0.0) for var  in self.binvars])
 

        self.c.variables.set_upper_bounds([(var, float(val)) for var, val in zip (cluster, sol[cluster]) ])
        self.c.variables.set_lower_bounds([(var, float(val)) for var, val in zip (cluster, sol[cluster]) ])
        #for var in cluster:
        #    model.variables.set_lower_bounds(var, float(sol[var]))
            



		# # model.start = var_starts
		# model.verbose = False

    #     start_time = self.c.get_time()
    #      end_time = c.get_time()
    # print("Total solve time (sec.):", end_time - start_time)
    # print("Network time (sec.):", outproc.network_time)

        start_time = time.time()
        self.c.parameters.mip.limits.solutions.set(9223372036800000000) 
        status = self.c.solve()
        print('[solve by cluster]')
        print('solution cost {} found'.format(self.c.solution.get_objective_value()))
        print ("MIP Status: ", status)
        print ("MIP relative Gap: ", self.c.solution.MIP.get_mip_relative_gap())

        end_time = time.time()
        run_time = end_time - start_time

        new_sol = np.zeros(len(self.binvars))

        new_sol = [round(self.c.solution.get_values(var))  for var in self.binvars ]

		# 		return sol, run_time, -1

        return np.array(new_sol), run_time, self.c.solution.get_objective_value(), status



    def optimize(self, init_sol = False):
        sol = []
        self.status = []
        self.sol_vals = []

        if init_sol :
            #self.c.parameters.parameters.mip.limits.solutions.set(1)
            self.c.parameters.mip.limits.solutions.set(1)
        else:
#            self.c.parameters.parameters.mip.limits.solutions.set(9223372036800)
            self.c.parameters.mip.limits.solutions.set(9223372036800000000) 

        # Solve the model.
        s = self.c.solve()
        status = self.c.solution.status[1] #self.c.solution.MIP.get_subproblem_status()
        print ("MIP Status: ", status)
        print ("MIP relative Gap: ", self.c.solution.MIP.get_mip_relative_gap())

        #self.status == self.c.get_status_string()
        self.obj_val = -1

        if status == 'optimal':
            print('optimal solution cost {} found'.format(self.c.solution.get_objective_value()))
        elif status == 'feasible ':
            print('sol.cost {} found, best possible: {}'.format(self.c.solution.get_objective_value()))
        elif status == 'infeasible':
            print('no feasible solution found, lower bound is: {}'.format(self.c.solution.get_objective_value()))

        sol = np.zeros(len(self.binvars))
        if status == 'optimal' or self.status == 'feasible' :
            if(init_sol):
                self.sol_vals = self.c.solution.get_values(self.binvars)
                self.obj_val = self.c.solution.get_objective_value()
                self.status = status
            sol = [round(self.c.solution.get_values(var))  for var in self.binvars ]
            
            #sol = round(self.c.solution.get_values(self.binvars))
        self.c.parameters.mip.limits.solutions.set(9223372036800000000) 
        

        return np.array(sol), self.c.solution.get_objective_value(), status


'''
dats = DATS_CPLEX("DATS","DATS/polska_01.lp")
clusters = dats.uniform_random_clusters(4)
sol,obj, status = dats.optimize(True)
#new_sol, run_time, objective_value = 
dats.solve_fixed_by_cluster(copy.copy(dats.c), clusters[0], sol )
'''
#print(new_sol, 0, objective_value)
#parameters.mip.limits.solutions
#>>> cpx.parameters.lpmethod.set(cpx.parameters.lpmethod.values.primal)
'''
inst = DATS_instance()
'''