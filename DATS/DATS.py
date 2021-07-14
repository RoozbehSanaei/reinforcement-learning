from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool
from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus, Column
import logging
from pprint import pprint
import numpy as np

from DATS_instance import *




class DATS:
	def __init__(self, _inst, lpfile, modelname):
		
		self.inst = _inst
		self.m = Model(modelname)
		self.m.read(lpfile)

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
	
	def init_vars(self):

		self.ServiceTime = self.inst.m_ServiceTime
		self.DockingTime = self.inst.m_DockingTime
		
		self.J0 = range (self.inst.m_nbTrucks)
		self.I0 = range (self.inst.m_nbTrucks)
		self.J = range (1, self.inst.m_nbTrucks)
		self.I = range (1, self.inst.m_nbTrucks)
		self.K = range (self.inst.m_nbDocks)
		self.T = range (self.inst.m_nbTF)
		self.S = range (self.inst.m_nbScenarios)

		self.z = [self.m.add_var(var_type=BINARY, name = "z("+ str(i) + ")") for i in range( self.inst.m_nbTrucks)]
		self.x = [[ [ [ None for t in self.T] for k in self.K] for j in self.J0] for i in self.I0]
		self.is_var = [[[False for t in self.T] for j in self.J0] for i in self.I0]

		self.varnames = []
		self.var_dict = {}
		for i in self.I0:
			for j in self.J0:
				for k in self.K:
					for t in self.T:
						if self.inst.isVariableDefined(i, j, t, 0):
							self.x[i][j][k][t] = self.m.add_var(var_type=BINARY, name = "x(" + str(i) + ")(" + str(j) + ")(" + str(k) + ")(" + str(t) + ")")
							self.is_var[i][j][t] = True
							#self.varnames.append("x(" + str(i) + ")(" + str(j) + ")(" + str(k) + ")(" + str(t) + ")")
							#self.var_dict['']

	def optimize(self):
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
				if abs(v.x) > 1e-6: # only printing non-zeros
					print('{} : {}'.format(v.name, v.x))





inst = DATS_instance ()
dats = DATS(inst, "Ok.lp", "DATS")
dats.optimize()
