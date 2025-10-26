import logging
from random import random
from typing import List, Dict, Any
from dataclasses import dataclass, field
import numpy as np


import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


from config.config import Config
from network import Node, Arc, TravelingArc, TransshipArc
from entity import Port, VesselType, VesselPath, ContainerPath, LadenPath, EmptyPath, ShipRoute, Request, ODRange

logger = logging.getLogger(__name__)

@dataclass
class DataManager:
    """
    输入数据类

    存储优化模型所需的所有输入数据

    主要数据分类:
    1. 基本参数: time_horizon, uncertain_degree等
    2. 网络数据: port_set, node_set, arc_set等
    3. 路径数据: vessel_path_set, laden_path_set, empty_path_set等
    4. 需求数据: request_set, sample_scenes等
    5. 历史数据: history_solution_set等
    """

    # 基本参数
    time_horizon: int = 0  # 时间范围
    uncertain_degree: float = 0.0  # 不确定度
    total_laden_paths_num: int = 0  # 重箱路径总数
    total_empty_paths_num: int = 0  # 空箱路径总数

    data = {}

    # 网络数据
    port_set: Dict[str, Port] = field(default_factory=dict)  # 港口集合
    ports: List[Port] = field(default_factory=list)  # 港口列表
    vessel_type_set: Dict[int, VesselType] = field(default_factory=dict)  # 船舶类型集合
    vessels: List[int] = field(default_factory=list)  # 船舶列表
    node_set: Dict[int, Node] = field(default_factory=dict)  # 节点集合
    nodes: List[Node] = field(default_factory=list)  # 节点列表
    arcs: List[Arc] = field(default_factory=list)  # 弧列表
    arc_set: Dict[int, Arc] = field(default_factory=dict)  # 弧集合
    arcs: List[Arc] = field(default_factory=list)  # 弧列表
    traveling_arcs: List[TravelingArc] = field(default_factory=list)  # 运输弧集合
    traveling_arc_set: Dict[int, TravelingArc] = field(default_factory=dict)  # 运输弧集合
    transship_arcs: List[TransshipArc] = field(default_factory=list)  # 转运弧集合
    transship_arc_set: Dict[int, TransshipArc] = field(default_factory=dict)  # 转运弧集合

    # 路径数据
    vessel_paths: List[VesselPath] = field(default_factory=list)  # 船舶路径集合
    vessel_path_set: Dict[int, VesselPath] = field(default_factory=dict)  # 船舶路径集合
    laden_paths: List[LadenPath] = field(default_factory=list)  # 重箱路径集合
    laden_path_set: Dict[int, LadenPath] = field(default_factory=dict)  # 重箱路径集合
    empty_paths: List[EmptyPath] = field(default_factory=list)  # 空箱路径集合
    empty_path_set: Dict[int, EmptyPath] = field(default_factory=dict)  # 空箱路径集合
    ship_routes: List[ShipRoute] = field(default_factory=list)  # 航线集合
    ship_route_set: Dict[int, ShipRoute] = field(default_factory=dict)  # 航线集合
    container_paths: List[ContainerPath] = field(default_factory=list)  # 集装箱路径集合
    container_path_set: Dict[int, ContainerPath] = field(default_factory=dict)  # 集装箱路径映射

    # 需求数据
    requests: List[Request] = field(default_factory=list)  # 需求请求集合
    request_set: Dict[int, Request] = field(default_factory=dict)  # 需求请求集合
    sample_scenes: Dict[int, List[float]] = field(default_factory=dict)  # 样本场景
    # 分组数据
    group_range_map: Dict[str, ODRange] = field(default_factory=dict)  # 起终点分组范围映射

    # 历史数据
    history_solution_set: Dict[str, List[int]] = field(default_factory=dict)  # 历史解决方案集合


    def get_group_range(self, origin_group: int, destination_group: int) -> ODRange:
        """
        获取起终点分组范围

        Args:
            origin_group: 起点分组
            destination_group: 终点分组

        Returns:
            ODRange: 起终点分组范围
        """
        key = f"{origin_group}{destination_group}"
        if key not in self.group_range_map:
            logger.info(f"group_range_map not found for key: {key}")
            raise ValueError(f"group_range_map not found for key: {key}")
        res = self.group_range_map.get(key)
        if res is None:
            logger.info(f"group_range_map get None for key: {key}")
            raise ValueError(f"group_range_map get None for key: {key}")
        return res

    def show_status(self):
        """
        显示数据状态
        """
        logger.info(f"\nTimeHorizon : {self.time_horizon}\n")
        logger.info(f"UncertainDegree : {self.uncertain_degree}\n")
        logger.info(
            f"Nodes = {len(self.node_set)}\t"
            f"TravelingArcs = {len(self.traveling_arc_set)}\t"
            f"TransshipArcs = {len(self.transship_arc_set)}\t\n"

            f"ShipRoute = {len(self.ship_route_set)}\t"
            f"Ports = {len(self.port_set)}\t"
            f"VesselPaths = {len(self.vessel_path_set)}\t"
            f"VesselTypes = {len(self.vessel_type_set)}\t\n"

            f"Requests = {len(self.request_set)}\t"
            f"Paths = {len(self.container_paths)}\t"
        )

        self.show_path_status()


    def show_path_status(self):
        """
        显示路径状态
        """
        total_laden_paths = 0
        total_empty_paths = 0

        for request in self.requests:
            total_laden_paths += request.number_of_laden_path
            total_empty_paths += request.number_of_empty_path

        logger.info(
            f"Requests = {len(self.request_set)}\t"
            f"Paths = {len(self.container_paths)}\t"
            f"Total LadenPaths = {total_laden_paths}\t"
            f"Total EmptyPaths = {total_empty_paths}"
        )

        self.total_laden_paths_num = total_laden_paths
        self.total_empty_paths_num = total_empty_paths


    def generate_demand_and_price(self):
        # 设置需求、惩罚成本等
        for request in self.requests:
            try:
                group_o = self.port_set[request.origin_port].group
                group_d = self.port_set[request.destination_port].group
                group_range = self.group_range_map[f"{group_o}{group_d}"]

                # 生成需求
                if group_range is not None:
                    demand_value = group_range.demand_lower_bound + int(
                        (group_range.demand_upper_bound - group_range.demand_lower_bound) *
                        random()
                    )
                    # 生成惩罚成本
                    price = group_range.freight_lower_bound + int(
                        (group_range.freight_upper_bound - group_range.freight_lower_bound) *
                        random()
                    )
                else:
                    demand_value = 0
                    price = 0

                demand_maximum_value = demand_value * self.uncertain_degree

                request.long_haul_price = price * Config.LONG_HAUL_COEFFICIENT
                request.penalty_cost = price * Config.PENALTY_COEFFICIENT
                request.mean_demand = demand_value
                request.base_demand = demand_value
                request.variance_demand = demand_maximum_value

            except Exception as e:
                logger.error(f"Error in set_requests: {e}")

    @property
    def vessel_types(self) -> List[VesselType]:
        return list(self.vessel_type_set.values())

    @property
    def shipping_routes(self) -> List[ShipRoute]:
        return list(self.ship_route_set.values())
