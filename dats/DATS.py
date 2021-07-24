from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool
from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus, Column
import logging
from pprint import pprint
import numpy as np
import random
import time 
import copy
import os

class truck:
	def __init__(self, i, _arrival, _serviceTime, _dockingTime, _deadline, _weighedCostCoef,_resourcePersonel, _resourceEquipment, _resourceVehicle ):
		self.arrivalTime = _arrival
		self.dockingTime = _dockingTime
		self.departureTime = _deadline
		self.weightedCostCoef = _weighedCostCoef

		self.serviceTime = _serviceTime
		self.resourcePersonel = _resourcePersonel
		self.resourceEquipments = _resourceEquipment
		self.resourceVehicle = _resourceVehicle


class DATS_instance:
	def __init__(self ):
		self.m_nbDocks = -1
		self.m_nbTrucks = -1
		self.m_nbTF = -1
		self.m_nbScenarios = -1
		self.m_nbResourcePersonel = -1
		self.m_nbResourceEquipment  = -1
		self.m_nbResourceVehicule = -1
		self.m_Arrival = []
		self.m_Deadline = []
		self.m_DockingTime = []
		self.m_WeighedCostCoef = []

		# 2D arrays
		self.m_ServiceTime = [[]]
		self.m_ResourcePersonel = [[]]
		self.m_ResourceEquipments = [[]]
		self.m_ResourceVehicle = [[]]

		self.trucks = []


	def isVariableDefined(self,  i,  j,  t,  s):

		if i == 0 and j >= 1  and t >= self.m_Arrival[j] + 1:
			return False
		if j == 0 and i >= 1 and t > self.m_Deadline[i]:
			return False

		if i == j and i > 0 and j > 0:
			return False
		if i > 0:
			if t > self.m_Deadline[j]:
				return False
			#if (t > m_Deadline()[i])
			#    return False;
			if t < self.m_Arrival[j]:
				return False
			if t < self.m_Arrival[i] + self.m_ServiceTime[i][s] + self.m_DockingTime[i]: 
				return False
			if t + self.m_ServiceTime[j][s] + self.m_DockingTime[j] > self.m_Deadline[j]:
				return False
			if t + self.m_ServiceTime[j][s] + self.m_DockingTime[j] > self.get_nbTF() - 1:
				return False
		
		else:
			if t > self.m_Deadline[j] or t < self.m_Arrival[j] or t + self.m_ServiceTime[j][s] + self.m_DockingTime[j] > self.m_Deadline[j] \
			or t + self.m_ServiceTime[j][s] + self.m_DockingTime[j] > self.m_nbTF - 1:
				return False
		return True


class DATS:
	def __init__(self, _inst,model_name,lp_file_name):
		self.inst = _inst
		self.m = Model(model_name)
		self.m.read(lp_file_name)


		# read all constraint and create 2 dicts
		self.constr_dict_index_name = {}
		self.constr_dict_index_var = {}
		self.constr_dict_name_index = {}

		self.constr_count = -1
		for c in self.m.constrs:
			self.constr_count += 1
			self.constr_dict_index_name[self.constr_count] = c.name
			self.constr_dict_name_index[c.name] = self.constr_count
			self.constr_dict_index_var[self.constr_count] = c

		self.var_dict = {}
		self.var_dict_index_name = {}
		self.var_dict_index_var = {}
		self.var_dict_name_index = {}

		self.var_in_constrs_index = {}
		self.var_in_constrs_name = {}
		self.var_coeffs_in_constrs_index = {}
		self.var_count = -1
		for v in self.m.vars:
			self.var_count += 1
			self.var_dict_index_name[self.var_count] = v.name
			self.var_dict_index_var[self.var_count] = v
			self.var_dict_name_index[v.name] = self.var_count

			self.var_in_constrs_name[self.var_count] = [e.name for e in v.column.constrs or [] if (v.column.coeffs) is not None and v.column.constrs is not None]
			self.var_in_constrs_index[v.name] = [self.constr_dict_name_index[e.name] for e in v.column.constrs or []]
			self.var_coeffs_in_constrs_index[v.name] = [e for e in v.column.coeffs or []]
			
		#print(self.var_in_constrs_name)
		#print(self.var_in_constrs_index)

		self.adjacency = np.zeros((len(self.var_dict_name_index), len(self.constr_dict_name_index)))

		for key in self.var_dict_name_index:
			for i in range(len (self.var_in_constrs_index[key])):
				#print(self.var_in_constrs_index[key][i] , self.var_coeffs_in_constrs_index[key][i])
				self.adjacency[self.var_dict_name_index[key]][self.var_in_constrs_index[key][i] ] = self.var_coeffs_in_constrs_index[key][i]

	
	def uniform_random_clusters(self,  num_clusters):
		'''Return a random clustering. Each node is assigned to a cluster
		a equal probability.'''

		choices = list(range(num_clusters))
		clusters = dict([(i, []) for i in range(num_clusters)])

		for k in self.var_dict_index_name.keys():
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

		var_starts = []
		for i in range(len(self.var_dict_name_index)):
			var = self.var_dict_index_var[i]
			if i in cluster:
				var_starts.append((var, sol[i]))
			else:
				model += var == sol[i]

		# for k, varname in self.var_dict_index_name.items():
		# 		# warm start variables in the current coordinate set with the existing solution.
		# 		model_var = model.var_by_name(varname)
		# 		if k in cluster:
		# 			var_starts.append((model_var, sol[k]))
		# 		else:
		# 			model += model_var == sol[k]

		# model.start = var_starts
		model.verbose = False
		start_time = time.time()
		model.optimize()
		end_time = time.time()
		run_time = end_time - start_time
		new_sol = np.zeros(len(self.var_dict_name_index))


		# for k, var in self.var_dict_index_name.items():
		# 	var = model.var_by_name(varname)
		# 	try:
		# 		new_sol.append( round(var.x))
		# 	except:
		# 		return sol, run_time, -1

		for i in range(len(self.var_dict_name_index)):
			var = self.var_dict_index_var[i]
			try:
				new_sol[i] =  round(var.x)
			except:
				return sol, run_time, -1


		return new_sol, run_time, model.objective_value

	def optimize(self):
		sol = []
		self.m.max_gap = 0.05
		#status = self.m.optimize(max_seconds=100)
		status = self.m.optimize(max_solutions = 1)
		if status == OptimizationStatus.OPTIMAL:
			print('optimal solution cost {} found'.format(self.m.objective_value))
		elif status == OptimizationStatus.FEASIBLE:
			print('sol.cost {} found, best possible: {}'.format(self.m.objective_value, self.m.objective_bound))
		elif status == OptimizationStatus.NO_SOLUTION_FOUND:
			print('no feasible solution found, lower bound is: {}'.format(self.m.objective_bound))
		if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
			print('solution:')
			for v in self.m.vars:
				#if abs(v.x) > 1e-6: # only printing non-zeros
					#print('{} : {}'.format(v.name, v.x))
				sol.append(round(v.x))
		return np.array(sol), self.m.objective_value

'''
inst = DATS_instance()
dats = DATS(inst,"DATS","DATS/polska_01.lp")
clusters = dats.uniform_random_clusters(4)
sol,obj = dats.optimize()
new_sol, run_time, objective_value = dats.solve_fixed_by_cluster(dats.m.copy(), clusters[0], sol )
print(new_sol, 0, objective_value)
'''