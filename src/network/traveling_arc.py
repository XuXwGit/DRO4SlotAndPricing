from dataclasses import dataclass
from .arc import Arc
from .node import Node

@dataclass
class TravelingArc(Arc):
    """
    运输弧类
    
    继承自Arc类，表示运输网络中的运输弧
    
    属性:
        traveling_arc_id: 运输弧ID
        route_id: 航线ID
        round_trip: 往返次数
        travel_time: 运输时间
    """
    def __init__(self, 
                 arc_id: int, 
                 route_id: int, 
                 round_trip: int, 
                 travel_time: int, 
                 origin_node: Node, 
                 destination_node: Node
                 ):
        
        super().__init__(arc_id, 
                         origin_node, 
                         destination_node, 
                        )
        self.traveling_arc_id = arc_id
        self.route_id = route_id
        self.round_trip = round_trip
        self.travel_time = travel_time

    traveling_arc_id: int  # 运输弧ID
    route_id: int  # 航线ID
    round_trip: int  # 往返次数
    travel_time: int  # 运输时间 