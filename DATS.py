from mip import Model, xsum, BINARY, minimize, ConstrsGenerator, CutPool
from mip import Model, MAXIMIZE, CBC, INTEGER, OptimizationStatus, Column
from itertools import product
import logging
from pprint import pprint
import numpy as np

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
	def __init__(self, datfilename ):
		self.m_fname = datfilename
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



	def parse(self):
		f = open(self.m_fname, "r")
		while 1:
			line = f.readline()
			if "end{Tasks}" in line:
				break

			if 'begin{nbTrucks}' in line:
				self.m_nbTrucks = int(f.readline())
				self.m_nbTrucks += 1

			if 'begin{nbDocks}' in line:
				self.m_nbDocks = int(f.readline())

			if 'begin{nbTF}' in line:
				self.m_nbTF = int(f.readline())

			if 'begin{nbScenarios}' in line:
				self.m_nbScenarios = int(f.readline())				
				

			if 'begin{nbResourcePersonel}' in line:
				self.m_nbResourcePersonel = int(f.readline())				

			if 'begin{nbResourceEquipment}' in line:
				self.m_nbResourceEquipment = int(f.readline())				

			if 'begin{nbResourceVehicule}' in line:
				self.m_nbResourceVehicule = int(f.readline())				

			if 'begin{Tasks}' in line:
				self.m_ServiceTime = [[] for i in range(self.m_nbTrucks)]
				self.m_ResourcePersonel = [[] for i in range(self.m_nbTrucks)]
				self.m_ResourceEquipments = [[] for i in range(self.m_nbTrucks)]
				self.m_ResourceVehicle = [[] for i in range(self.m_nbTrucks)]

				self.m_Arrival.append(0)
				self.m_Deadline.append(self.m_nbTF)
				self.m_DockingTime.append(0)
				self.m_WeighedCostCoef.append(0)

				for s in range(self.m_nbScenarios):
					self.m_ServiceTime[0].append(0)
					self.m_ResourcePersonel[0].append(0)
					self.m_ResourceEquipments[0].append(0)
					self.m_ResourceVehicle[0].append(0)


				for i in range(1, self.m_nbTrucks):
					line = f.readline()
					vals = line.split()
					self.m_Arrival.append(int(vals[1]))
					self.m_Deadline.append(int(vals[2]))
					self.m_DockingTime.append(int(vals[3]))
					self.m_WeighedCostCoef.append(int(vals[4]))
					
					cntr = 4
					for s in range(self.m_nbScenarios):
						cntr +=1
						self.m_ServiceTime[i].append(int(vals[cntr]))
						cntr +=1
						self.m_ResourcePersonel[i].append(int(vals[cntr]))
						cntr +=1
						self.m_ResourceEquipments[i].append(int(vals[cntr]))
						cntr +=1
						self.m_ResourceVehicle[i].append(int(vals[cntr]))
						

					trc = truck (i, self.m_Arrival[i], self.m_ServiceTime[i], self.m_DockingTime[i], self.m_Deadline[i], self.m_WeighedCostCoef[i], \
						self.m_ResourcePersonel[i], self.m_ResourceVehicle[i], self.m_ServiceTime[i])
					self.trucks.append(trc)
				# 		//trc.arrivalTime = m_Arrival[i];
				# 		//trc.departureTime = m_Deadline[i];
				# 		//trc.serviceTime = m_ServiceTime[i];
				# 		//trc.dockingTime = m_DockingTime[i];
				# 		//trc.id = i;

				# 		//trucks_by_arrivalTime.insert(pair<int, truck>(m_Arrival[i], trc));
				# 		//trucks_by_departureTime.insert(pair<int, truck>(m_Deadline[i], trc));
				# 		//trucks_by_serviceTime.i nsert(pair<int, truck>(m_ServiceTime[i], trc));
				# 		//trucks_by_dockingTime.insert(pair<int, truck>(m_DockingTime[i], trc));
				# 		//trucks_by_weightedCostCoef.insert(pair<int, truck>(m_WeighedCostCoef[i], trc));
				# 		//trucks_by_DockingPlusServiceTime_less.insert(pair<int, truck>(m_DockingTime[i]+ m_ServiceTime[i], trc));
				# 		//trucks_by_DockingPlusServiceTime_more.insert(pair<int, truck>(m_DockingTime[i] + m_ServiceTime[i], trc));
				# 		//trucks_by_arrivalPlusServicePlusDockingTime.insert(pair<int, truck>(m_Arrival[i] + m_DockingTime[i] + m_ServiceTime[i], trc));

				# 		//ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				# 	}
				# }
				





		return 



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






	def getArrival(self):
		return self.m_Arrival

	def getDeadline(self):
		return self.m_Deadline	

	def getDockingTime(self):
		return self.m_DockingTime

	def getWeightedCostCoef(self):
		return self.m_WeighedCostCoef
	#get 2D 
	def getProcessingTime(self):
		return self.m_ServiceTime

	def getRequiredResourcePersonel(self):
		return self.m_ResourcePersonel

	def getRequiredResourceEquipments(self):
		return self.m_ResourceEquipments

	def getRequiredResourceVehicle(self):
		return self.m_ResourceVehicle

	#get scalars

	def get_nbTrucks(self):
		return self.m_nbTrucks


	def get_nbDocks(self):
		return self.m_nbDocks

	def get_nbTF(self):
		return self.m_nbTF

	def get_nbScenarios(self):
		return self.m_nbScenarios


class DATS:
	def __init__(self, _inst, lpfile, modelname):
		
		self.inst = _inst
		self.m = Model("DATS")
		self.m.read('Ok.lp')

		# read all constraint and create 2 dicts
		self.constr_dict_index = {}
		self.constr_dict_name = {}
		self.constr_count = -1
		for c in self.m.constrs:
			self.constr_count += 1
			self.constr_dict_index[self.constr_count] = c.name
			self.constr_dict_name[c.name] = self.constr_count

		self.var_dict = {}
		self.var_dict_name = {}
		self.var_dict_index = {}
		self.var_in_constrs_index = {}
		self.var_in_constrs_name = {}
		self.var_coeffs_in_constrs_index = {}
		self.var_count = -1
		for v in self.m.vars:
			self.var_count += 1
			self.var_dict_index[self.var_count] = v.name
			self.var_dict_name[v.name] = self.var_count

			self.var_in_constrs_name[self.var_count] = [e.name for e in v.column.constrs or [] if (v.column.coeffs) is not None and v.column.constrs is not None]
			self.var_in_constrs_index[v.name] = [self.constr_dict_name[e.name] for e in v.column.constrs or []]
			self.var_coeffs_in_constrs_index[v.name] = [e for e in v.column.coeffs or []]
			
		#print(self.var_in_constrs_name)
		#print(self.var_in_constrs_index)

		self.adjacency = np.zeros((len(self.var_dict_name), len(self.constr_dict_name)))

		for key in self.var_dict_name:
			for i in range(len (self.var_in_constrs_index[key])):
				#print(self.var_in_constrs_index[key][i] , self.var_coeffs_in_constrs_index[key][i])
				self.adjacency[self.var_dict_name[key]][self.var_in_constrs_index[key][i] ] = self.var_coeffs_in_constrs_index[key][i]
	
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




	def build_model(self, export_lp = False):
		

		self.s = 0
	 	#	flow conservation at existing nodes

		#for i in self.I:
		#	self.m += xsum(self.x[j][i][k][t] for  j in self.J0 for  k in self.K for t in self.T if self.inst.isVariableDefined(i,  j,  t,  0 ) and i != j)  + self.z[i] == 1 , "rng1(" + str(i)+ ")"
		print("RNG1")
		for i in self.I:
			expr =  self.z[i] 
			for j,k,t in product(self.J0, self.K, self.T):
				if self.is_var[i][j][t] and i != j:
					expr = expr + self.x[i][j][k][t] 
			expr = expr == 1
			self.m += expr , "rng1(" + str(i)+ ")"

		
		print("RNG2")
		for i in self.I:
			self.m += xsum(self.x[i][j][k][t] for  j in self.J0 for  k in self.K for t in self.T if self.is_var[i][j][t] and i != j)   == 1, "rng2(" + str(i)+ ")"

		# # *	outgoing  + z  == 1
		# # *	incomming  + z  == 1
		print("RNG3")
		for k in self.K:
			self.m += xsum(self.x[0][j][k][t] for  j in self.J  for t in self.T if self.is_var[0][j][t] )  + self.x[0][0][k][0] == 1, "rng3(" + str(k)+ ")"
		# for  k in self.K:
		# 	expr =  self.x[0][0][k][0]
		# 	for j in self.J:
		# 		for t in self.T:
		# 			if self.inst.isVariableDefined(0,  j,  t,  0 ) :
		# 				expr = expr + self.x[0][j][k][t]
		# 	expr = expr == 1
		# 	self.m += expr , "rng3(" + str(i)+ ")"
		print("RNG4")
		for k in self.K:
			self.m += xsum(self.x[i][0][k][t] for  j in self.J  for t in self.T if self.is_var[i][0][t] )  + self.x[0][0][k][0] == 1, "rng4(" + str(k)+ ")"
		
		print("RNG5")					
		# *	X	<	nextX
		for k, i, j, t in product(self.K, self.I0, self.J, self.T):
			if self.is_var[i][j][t] and j!=i:
				flag = False
				for l in self.I0:
				 	if ((l != i or l == 0) and j != l):
				 		if l == 0:
				 			tt = t + self.ServiceTime[j][self.s] + self.DockingTime[j]
				 			expr = expr +  self.x[j][l][k][tt]
				 			flag = True
				# 		else:
		
				# 			idx = max([t, self.inst.m_Arrival[l], self.inst.m_Arrival[j] + self.inst.m_ServiceTime[j][self.s] + self.inst.m_DockingTime[j], \
				# 					t + self.inst.m_ServiceTime[j][self.s] + self.inst.m_DockingTime[j] ])
		
				# 			for  tt  in range(idx,  self.inst.m_nbTF):
				# 				if self.is_var[j][l][tt] :
				# 					expr =  expr + self.x[j][l][k][tt]
				# 					flag = True
				# 					break
				if flag :
					self.m =   self.x[i][j][k][t] - expr <= 0#,  "rng5(" + str(i) + ")(" + str(j) + ")(" + str(t) + ")"
				else:
					self.x[i][j][k][t].ub = 0
		print("RNG6")					
		# *	Variable Fixing
		cntr = 0
		for k, j in product(self.K, self.J):
			for t in range( self.inst.m_Arrival[j] + 1,  self.inst.m_nbTF):
				if self.is_var[0][j][t]:
					self.x[0][j][k][t].ub = 0
					cntr += 1


		#objective function

		self.obj = xsum( self.z[j] *  inst.getWeightedCostCoef()[j] * 100 for j in self.J)

		for k,i,j,t in product(self.K, self.I0, self.J, self.T):
			if self.inst.isVariableDefined(i, j, t, 0):
				self.obj = self.obj + self.x[i][j][k][t] * (t - inst.getArrival()[j]) * inst.getWeightedCostCoef()[j]

		

		self.m.objective = minimize(self.obj)

		if export_lp:
			self.m.write('model.lp')

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
''''
bool getMinMaxTrucks(int& _min, int& _max);
'''''






inst = DATS_instance ("D:\\Work\\projects\\cross_dock_2020\\DATS\\code\\generator\\instances\\RC\\1-Scenario\\tf-16-d-20-tr-65-Sce-1-RC.dat")
inst.parse()
dats = DATS(inst, "Ok.lp", "DATS")
dats.optimize()
