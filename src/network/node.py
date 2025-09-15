from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


from src.entity.port import Port

@dataclass
class Node:
    """
    网络节点类，表示运输网络中的一个节点。
    对应Java类: multi.network.Node
    """
    
    def __init__(self, 
                 id: int = 0, 
                 route: int = 0, 
                 call: int = 0, 
                 port_string: str = "", 
                 round_trip: int = 0, 
                 time: int = 0, 
                 port: Optional[Port] = None):
        """
        初始化节点对象
        
        Args:
            port_string: 港口字符串，默认为空字符串
            id: 节点ID，默认为0
            route: 航线，默认为0
            call: 停靠次数，默认为0
            round_trip: 往返次数，默认为0
            time: 时间点，默认为0
            port: 关联的港口对象，默认为None
        """
        # 基本属性，对应Java类中的字段
        self.port_string = port_string  # 对应Java: private String portString
        self.node_id = id  # 对应Java: private int nodeID
        self.route = route  # 对应Java: private int route
        self.call = call  # 对应Java: private int call
        self.round_trip = round_trip  # 对应Java: private int roundTrip
        self.time = time  # 对应Java: private int time
        self.port = port  # 对应Java: private Port port
        
        # 附加属性（Python实现中用于网络构建）
        self.incoming_arcs: List[Any] = []  # 入边列表
        self.outgoing_arcs: List[Any] = []  # 出边列表
    
    def add_incoming_arc(self, arc: Any):
        """
        添加一条入边到节点
        
        Args:
            arc: 要添加的弧对象
        """
        self.incoming_arcs.append(arc)

    def add_outgoing_arc(self, arc: Any):
        """
        添加一条出边到节点（Python特有方法，用于网络构建）
        
        Args:
            arc: 要添加的弧对象
        """
        self.outgoing_arcs.append(arc)

    def __str__(self):
        """
        返回节点的字符串表示
        
        Returns:
            str: 节点的字符串描述
        """
        return f"Node(id={self.node_id}, port={self.port_string}, time={self.time})" 