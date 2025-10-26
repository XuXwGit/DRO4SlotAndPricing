"""
@Author: XuXw
@Description: 订单/需求 类
@DateTime: 2024/12/4 21:31
"""
import logging
from typing import Dict, List, Optional
from .port import Port
from .container_path import ContainerPath

# 设置日志
logger = logging.getLogger(__name__)


class Request:
    """
    订单/需求类
    
    该类包含了运输请求的所有相关信息，包括：
    - 请求基本信息（ID、到达时间等）
    - 需求统计信息（均值、方差）
    - 起终点信息（港口对象、港口名称、港口组）
    - 时间窗口（最早提货时间、最晚到达时间）
    - 路径信息（重箱路径和空箱路径）
    - 惩罚成本（未满足需求的惩罚）
    """
    
    def __init__(self, 
                 request_id: int = 0, 
                 origin_port: str = "", 
                 destination_port: str = "", 
                 earliest_pickup_time: int = 0, 
                 latest_destination_time: int = 0):
        """
        初始化Request对象
        
        Args:
            request_id: 请求ID，默认为0
            origin_port: 起始港口名称，默认为空字符串
            destination_port: 目标港口名称，默认为空字符串
            earliest_pickup_time: 最早提货时间，默认为0
            latest_destination_time: 最晚到达时间，默认为0
        """
        # 基本属性
        self.request_id = request_id  
        self.arrival_time = 0  
        
        # 需求相关属性
        self.base_demand = 0.0
        self.mean_demand = 0.0  
        self.variance_demand = 0.0  
        
        # 成本属性
        self.penalty_cost = 0.0
        self.long_haul_price = 100.0
        
        # 港口对象
        self.origin = None  
        self.destination = None  
        
        # 港口字符串表示
        self.origin_port = origin_port  
        self.destination_port = destination_port  
        self.origin_group = 0  
        self.destination_group = 0  
        self.earliest_pickup_time = earliest_pickup_time  
        self.latest_destination_time = latest_destination_time  
        
        # 重箱路径相关
        self.laden_path_set = {}  
        self.laden_paths = []  
        self.laden_path_indexes = []  
        self.number_of_laden_path = 0  
        
        # 空箱路径相关
        self.empty_path_set = {}  
        self.empty_paths = []  
        self.empty_path_indexes = []  
        self.number_of_empty_path = 0  
        
    def __str__(self) -> str:
        """
        返回请求的字符串表示
        
        Returns:
            str: 请求的字符串描述
        """
        return (f"Request(id={self.request_id}, "
                f"from={self.origin_port}, "
                f"to={self.destination_port}, "
                f"demand={self.mean_demand})")
                
    def add_laden_path(self, path: ContainerPath) -> None:
        """
        添加重箱路径
        
        Args:
            path: 要添加的重箱路径
        """
        self.laden_path_set[path.path_id] = path
        if hasattr(path, 'container_path_id'):
            self.laden_paths.append(path.container_path_id)
        else:
            # 如果没有container_path_id属性，尝试使用id属性
            self.laden_paths.append(path.id if hasattr(path, 'id') else 0)
        self.number_of_laden_path += 1

    def add_empty_path(self, path: ContainerPath) -> None:
        """
        添加空箱路径
        
        Args:
            path: 要添加的空箱路径
        """
        self.empty_path_set[path.id] = (path)
        if hasattr(path, 'container_path_id'):
            self.empty_paths.append(path.container_path_id)
        else:
            # 如果没有container_path_id属性，尝试使用id属性
            self.empty_paths.append(path.id if hasattr(path, 'id') else 0)
        self.number_of_empty_path += 1 