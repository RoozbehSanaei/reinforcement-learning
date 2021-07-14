
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
