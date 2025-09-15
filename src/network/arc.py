"""
@Author: xuxw
@Description: 网络弧类
@DateTime: 2024/12/18 16:30
"""
from typing import Optional
from network.node import Node  # 移除顶部import，延迟导入

class Arc:
    """
    弧/边类，表示网络中的连接
    对应Java类: multi.network.Arc
    """
    
    def __init__(self, 
                 arc_id, 
                 origin_node: Node,
                 destination_node: Node):
        """
        初始化Arc对象
        
        Args:
            arc_id: 弧的唯一标识符，默认为0
            origin_node: 起始节点，默认为None
            destination_node: 目标节点，默认为None
        """
        # 延迟导入，避免循环依赖
        # from multi.network.node import Node
        # 基本属性
        self.arc_id = arc_id  # 对应Java: private int arcID
        self.origin_node_id = origin_node.node_id  # 对应Java: private int originNodeID
        self.destination_node_id = destination_node.node_id  # 对应Java: private int destinationNodeID
        self.origin_port = origin_node.port  # 对应Java: private String originPort
        self.destination_port = destination_node.port  # 对应Java: private String destinationPort
        self.origin_node = origin_node  # 对应Java: private Node originNode
        self.destination_node = destination_node  # 对应Java: private Node destinationNode
        self.origin_call = origin_node.call  # 对应Java: private int originCall
        self.destination_call = destination_node.call  # 对应Java: private int destinationCall
        self.origin_time = origin_node.time  # 对应Java: private int originTime
        self.destination_time = destination_node.time  # 对应Java: private int destinationTime
        
        # 弧类型："Traveling Arc" or "Transship Arc"
        self.arc_type = ""  # 对应Java: private String arcType
        
        # 若节点不为空，将弧添加到节点的边集合中
        if origin_node:
            origin_node.add_outgoing_arc(self)
        if destination_node:
            destination_node.add_incoming_arc(self)

    
    def __str__(self):
        """
        返回弧的字符串表示
        
        Returns:
            str: 弧的字符串描述
        """
        return f"Arc(id={self.arc_id}, from={self.origin_node_id}, to={self.destination_node_id})" 