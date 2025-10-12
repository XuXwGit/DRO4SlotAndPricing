"""
@Author: XuXw
@Description: 转运弧类，继承自Arc类
@DateTime: 2024/12/4 21:54
"""
from typing import Optional
from .arc import Arc
from .node import Node


class TransshipArc(Arc):
    """
    转运弧类，继承自Arc
    """
    
    def __init__(self, 
                 id: int, 
                 port: str,
                 origin_node: Node, 
                 destination_node: Node,
                 from_route: int = 0,
                 to_route: int = 0,
                 transship_time: int = 0,
                 ):
        """
        初始化转运弧对象
        
        Args:
            arc_id: 弧的唯一标识符，默认为0
            origin_node: 起始节点，默认为None
            destination_node: 目标节点，默认为None
        """
        # 调用父类构造函数
        super().__init__(id, origin_node, destination_node)
        
        # 转运弧特有属性
        self.transship_arc_id = id  
        self.port = port  
        self.transship_time = transship_time  
        self.from_route = from_route  
        self.to_route = to_route  
        
    def __str__(self):
        """
        返回转运弧的字符串表示
        
        Returns:
            str: 转运弧的字符串描述
        """
        return f"TransshipArc(id={self.arc_id}, port={self.port}, from_route={self.from_route}, to_route={self.to_route})" 