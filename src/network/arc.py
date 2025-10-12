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
        self.arc_id = arc_id
        self.id = arc_id
        self.capacity = 5000
        self.origin_node_id = origin_node.node_id
        self.destination_node_id = destination_node.node_id
        self.origin_port = origin_node.port
        self.destination_port = destination_node.port 
        self.origin_node = origin_node
        self.destination_node = destination_node  
        self.origin_call = origin_node.call  
        self.destination_call = destination_node.call  
        self.origin_time = origin_node.time  
        self.destination_time = destination_node.time
        
        # 弧类型："Traveling Arc" or "Transship Arc"
        self.arc_type = ""  
        
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