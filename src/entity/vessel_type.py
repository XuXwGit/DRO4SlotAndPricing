"""
@Author: XuXw
@Description: 船舶类型类，对应Java版本VesselType.java
@DateTime: 2024/12/4 21:54
"""


class VesselType:
    """
    船舶类型类
    对应Java类: multi.network.VesselType
    
    存储船舶类型相关的属性和信息
    """
    
    def __init__(self, id: int = 0, capacity: int = 0, cost: float = 0.0, route_id: int = 0, max_num: int = 0):
        """
        初始化船舶类型对象
        
        Args:
            id: 船舶类型ID，默认为0
            capacity: 容量，默认为0
            cost: 成本，默认为0.0
            route_id: 航线ID，默认为0
            max_num: 最大数量，默认为0
        """
        # 基本属性
        self._id = id  # 对应Java: private int id
        self._capacity = capacity  # 对应Java: private int capacity
        self._cost = cost  # 对应Java: private double cost
        self._route_id = route_id  # 对应Java: private int routeID
        self._max_num = max_num  # 对应Java: private int maxNum
    
    # Getter和Setter方法
    @property
    def id(self) -> int:
        """
        获取船舶类型ID
        对应Java: getId()
        """
        return self._id
    
    @id.setter
    def id(self, value: int):
        """
        设置船舶类型ID
        对应Java: setId(int id)
        """
        self._id = value
    
    @property   
    def vessel_id(self) -> int:
        """
        获取船舶ID
        对应Java: getVesselID()
        """
        return self._id


    @property
    def capacity(self) -> int:
        """
        获取容量
        对应Java: getCapacity()
        """
        return self._capacity
    
    @capacity.setter
    def capacity(self, value: int):
        """
        设置容量
        对应Java: setCapacity(int capacity)
        """
        self._capacity = value
    
    @property
    def cost(self) -> float:
        """
        获取成本
        对应Java: getCost()
        """
        return self._cost
    
    @cost.setter
    def cost(self, value: float):
        """
        设置成本
        对应Java: setCost(double cost)
        """
        self._cost = value
    
    @property
    def route_id(self) -> int:
        """
        获取航线ID
        对应Java: getRouteID()
        """
        return self._route_id
    
    @route_id.setter
    def route_id(self, value: int):
        """
        设置航线ID
        对应Java: setRouteID(int routeID)
        """
        self._route_id = value
    
    @property
    def max_num(self) -> int:
        """
        获取最大数量
        对应Java: getMaxNum()
        """
        return self._max_num
    
    @max_num.setter
    def max_num(self, value: int):
        """
        设置最大数量
        对应Java: setMaxNum(int maxNum)
        """
        self._max_num = value
        
    def __str__(self) -> str:
        """
        返回船舶类型的字符串表示
        
        Returns:
            str: 船舶类型的字符串描述
        """
        return f"VesselType(id={self.id}, capacity={self.capacity}, cost={self.cost}, route_id={self.route_id}, max_num={self.max_num})" 