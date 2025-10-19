from dataclasses import dataclass, field
from typing import List
from src.network.arc import Arc

@dataclass
class LadenPath:
    """
    重箱路径类

    存储重箱路径相关的属性和信息

    属性:
        container_path: 集装箱路径对象
        request_id: 请求ID
        origin_port: 起始港口
        origin_time: 起始时间
        destination_port: 终止港口
        round_trip: 往返次数
        earliest_setup_time: 最早设置时间
        arrival_time_to_destination: 到达目的地时间
        path_time: 路径时间
        transshipment_port: 转运港口列表
        transshipment_time: 转运时间列表
        port_path: 港口路径列表
        path_id: 路径ID
        number_of_arcs: 弧的数量
        arcs_id: 弧ID列表
        arcs: 弧对象列表
    """
    def __init__(self,
                 request_id: int,
                 origin_port: str,
                 origin_time: int,
                 destination_port: str,
                 round_trip: int,
                 earliest_setup_time: int,
                 arrival_time_to_destination: int,
                 path_time: int,
                 ports_in_path: List[str],
                 path_id: int,
                 arc_ids: List[int],
                 arcs: List[Arc]
                 ):
        self.request_id = request_id
        self.origin_port = origin_port
        self.origin_time = origin_time
        self.destination_port = destination_port
        self.round_trip = round_trip
        self.earliest_setup_time = earliest_setup_time
        self.arrival_time_to_destination = arrival_time_to_destination
        self.path_time = path_time
        self.ports_in_path =  ports_in_path
        self.path_id = path_id
        self.number_of_arcs = len(arc_ids)
        self.arc_ids = arc_ids
        self.arcs = arcs
