import truck

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
