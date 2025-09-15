from dataclasses import dataclass

@dataclass(frozen=True)
class ODRange:
    """
    起终点范围类
    
    存储起终点组对之间的需求和运费的上下界
    
    属性:
        origin_group: 起点组
        destination_group: 终点组
        demand_lower_bound: 需求下界
        demand_upper_bound: 需求上界
        freight_lower_bound: 运费下界
        freight_upper_bound: 运费上界
    """
    
    origin_group: int  # 起点组
    destination_group: int  # 终点组
    demand_lower_bound: int  # 需求下界
    demand_upper_bound: int  # 需求上界
    freight_lower_bound: int  # 运费下界
    freight_upper_bound: int  # 运费上界

    def __str__(self):
        """
        返回范围的字符串表示
        
        Returns:
            str: 范围的字符串描述
        """
        return (f"ODRange(origin={self.origin_group}, "
                f"destination={self.destination_group}, "
                f"demand=[{self.demand_lower_bound}, {self.demand_upper_bound}], "
                f"freight=[{self.freight_lower_bound}, {self.freight_upper_bound}])")

    def contains_demand(self, demand: int) -> bool:
        """
        检查给定的需求是否在范围内
        
        Args:
            demand: 要检查的需求值
            
        Returns:
            bool: 如果需求在范围内返回True，否则返回False
        """
        return self.demand_lower_bound <= demand <= self.demand_upper_bound

    def contains_freight(self, freight: int) -> bool:
        """
        检查给定的运价是否在范围内
        
        Args:
            freight: 要检查的运价值
            
        Returns:
            bool: 如果运价在范围内返回True，否则返回False
        """
        return self.freight_lower_bound <= freight <= self.freight_upper_bound 