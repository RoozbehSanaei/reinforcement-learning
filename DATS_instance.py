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
			print(line)

			if 'begin{nbTrucks}' in line:
				self.m_nbTrucks = int(f.readline())

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

				for i in range(self.m_nbTrucks):
					line = f.readline()
					vals = line.split()
					self.m_Arrival.append(int(vals[1]))
					self.m_Deadline.append(int(vals[2]))
					self.m_DockingTime.append(int(vals[3]))
					self.m_WeighedCostCoef.append(int(vals[4]))
					

					for s in range(self.m_nbScenarios):
						self.m_ServiceTime[i].append(int(vals[5]))
						self.m_ResourcePersonel[i].append(int(vals[6]))
						self.m_ResourceEquipments[i].append(int(vals[7]))
						self.m_ResourceVehicle[i].append(int(vals[8]))
						

					trc = truck (i, self.m_Arrival[i], self.m_ServiceTime[i], self.m_DockingTime[i], self.m_Deadline[i], self.m_WeighedCostCoef[i], \
						self.m_ResourcePersonel[i], self.m_ResourceVehicle[i], self.m_ServiceTime[i]);
					self.trucks.append(trc);
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
				if "end{Tasks}" in line:
					break





		return 



	def isVariableDefined(self,  i,  j,  t,  s):

			if i == 0 and j >= 1  and t >= m_Arrival[j] + 1:
				return False
			if j == 0 and i >= 1 and t > m_Deadline[i]:
				return False;

			if i == j and i > 0 and j > 0:
				return False
			if i > 0:
				if t > m_Deadline[j]:
					return False
				#if (t > m_Deadline()[i])
				#    return False;
				if t < m_Arrival[j]:
					return False
				if t < m_Arrival[i] + m_ServiceTime[i][s] + m_DockingTime[i]: 
					return False
				if t + m_ServiceTime[j][s] + m_DockingTime[j] > m_Deadline[j]:
					return False
				if t + m_ServiceTime[j][s] + m_DockingTime[j] > get_nbTF() - 1:
					return False
			
			else:
				if t > m_Deadline[j] or t < m_Arrival[j] or t + m_ServiceTime[j][s] + m_DockingTime[j] > m_Deadline[j] \
				or t + m_ServiceTime[j][s] + m_DockingTime[j] > m_nbTF - 1:
					return False







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

''''
bool getMinMaxTrucks(int& _min, int& _max);
'''''






inst = DATS_instance ("D:\\Work\\projects\\cross_dock_2020\\DATS\\code\\generator\\instances\\RC\\1-Scenario\\tf-16-d-20-tr-65-Sce-1-RC.dat")

inst.parse()