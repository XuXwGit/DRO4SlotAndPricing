"""
@Author: XuXw
@Description: 船舶类型类，
@DateTime: 2024/12/4 21:54
"""


class VesselType:
    """
    船舶类型类
    
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
        self._id = id  
        self._capacity = capacity  
        self._cost = cost  
        self._route_id = route_id  
        self._max_num = max_num  
    
    # Getter和Setter方法
    @property
    def id(self) -> int:
        """
        获取船舶类型ID
        
        """
        return self._id
    
    @id.setter
    def id(self, value: int):
        """
        设置船舶类型ID
        
        """
        self._id = value
    
    @property   
    def vessel_id(self) -> int:
        """
        获取船舶ID
        
        """
        return self._id


    @property
    def capacity(self) -> int:
        """
        获取容量
        
        """
        return self._capacity
    
    @capacity.setter
    def capacity(self, value: int):
        """
        设置容量
        
        """
        self._capacity = value
    
    @property
    def cost(self) -> float:
        """
        获取成本
        
        """
        return self._cost
    
    @cost.setter
    def cost(self, value: float):
        """
        设置成本
        
        """
        self._cost = value
    
    @property
    def route_id(self) -> int:
        """
        获取航线ID
        
        """
        return self._route_id
    
    @route_id.setter
    def route_id(self, value: int):
        """
        设置航线ID
        
        """
        self._route_id = value
    
    @property
    def max_num(self) -> int:
        """
        获取最大数量
        
        """
        return self._max_num
    
    @max_num.setter
    def max_num(self, value: int):
        """
        设置最大数量
        
        """
        self._max_num = value
        
    def __str__(self) -> str:
        """
        返回船舶类型的字符串表示
        
        Returns:
            str: 船舶类型的字符串描述
        """
        return f"VesselType(id={self.id}, capacity={self.capacity}, cost={self.cost}, route_id={self.route_id}, max_num={self.max_num})" 