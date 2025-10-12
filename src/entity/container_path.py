"""
@Author: XuXw
@Description: 集装箱路径类
@DateTime: 2024/12/4 21:54
"""
import logging
import os
from typing import List
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.entity.port import Port
from src.network.arc import Arc

# 设置日志
logger = logging.getLogger(__name__)


class ContainerPath:
    """
    集装箱路径类

    该类表示集装箱在网络中的运输路径，包含以下信息：
    - 路径基本信息（ID、起点、终点、时间等）
    - 转运港口和时间信息
    - 路径上的港口序列
    - 路径上的弧序列
    - 路径成本
    """

    def __init__(self,
                 container_path_id: int = 0,
                 origin_port: str = "",
                 destination_port: str = "",
                 origin_time: int = 0,
                 destination_time: int = 0,
                 ):
        """
        初始化ContainerPath对象

        Args:
            container_path_id: 集装箱路径ID，默认为0
            origin_port: 起始港口名称，默认为空字符串
            destination_port: 目标港口名称，默认为空字符串
            origin_time: 起始时间，默认为0
            destination_time: 目标时间，默认为0
        """
        # 基本属性
        self.container_path_id = container_path_id
        self.origin_port = origin_port
        self.origin_time = origin_time
        self.destination_port = destination_port
        self.destination_time = destination_time
        self.path_time = destination_time - origin_time

        # 转运信息
        self.transshipment_port: List[str] = []
        self.transshipment_time: List[int] = []
        self.total_transship_time = 0
        self.transshipment_ports: List[Port] = []

        # 路径信息
        self.number_of_path = 0
        self.port_path: List[str] = []
        self.ports_in_path: List[Port] = []

        # 弧信息
        self.number_of_arcs = 0
        self.arcs_id: List[int] = []
        self.arcs: List[Arc] = []

        # 成本
        self.path_cost = 0.0

    def get_total_transshipment_time(self) -> int:
        """
        计算总转运时间

        Returns:
            int: 总转运时间
        """

        total_transshipment_time = 0


        if not self.transshipment_port:

            self.total_transship_time = 0
            return 0


        for i in range(len(self.transshipment_port)):

            total_transshipment_time += self.transshipment_time[i]


        self.total_transship_time = total_transshipment_time


        return total_transshipment_time

    def get_total_demurrage_time(self) -> int:
        """
        计算总滞期时间


        Returns:
            int: 总滞期时间
        """

        total_transshipment_time = 0

        total_demurrage_time = 0


        if not self.transshipment_port:

            self.total_transship_time = 0

            return 0


        if len(self.transshipment_port) != len(self.transshipment_time):

            logger.info("Error in transshipment port num!")
            return 0


        for i in range(len(self.transshipment_port)):

            total_transshipment_time += self.transshipment_time[i]

            if self.transshipment_time[i] > 7:

                total_demurrage_time += (self.transshipment_time[i] - 7)


        self.total_transship_time = total_transshipment_time


        return total_demurrage_time


    @property
    def id(self) -> int:
        """
        获取路径ID

        """
        return self.container_path_id

    @property
    def path_id(self) -> int:
        """
        获取路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self.container_path_id

    @property
    def laden_path_id(self) -> int:
        """
        获取重箱路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self.container_path_id

    @property
    def empty_path_id(self) -> int:
        """
        获取空箱路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self.container_path_id


    @property
    def travel_time(self) -> int:
        """
        获取运输时间

        """
        return self.path_time


    @property
    def travel_time_on_path(self) -> int:
        """
        获取路径上的运输时间

        """
        return self.path_time



    @property
    def total_transship_time(self) -> int:
        """
        获取总转运时间

        """
        return self.total_transship_time

    @total_transship_time.setter
    def total_transship_time(self, value: int):
        """
        设置总转运时间

        """
        self.total_transship_time = value


    @property
    def number_of_path(self) -> int:
        """
        获取路径数量

        """
        return self.number_of_path

    @number_of_path.setter
    def number_of_path(self, value: int):
        """
        设置路径数量

        """
        self.number_of_path = value


    @property
    def number_of_arcs(self) -> int:
        """
        获取弧数量

        """
        return self.number_of_arcs

    @number_of_arcs.setter
    def number_of_arcs(self, value: int):
        """
        设置弧数量

        """
        self.number_of_arcs = value


    @property
    def cost(self) -> float:
        """
        获取路径成本

        """
        return self.path_cost


    @property
    def path_cost(self) -> float:
        """
        获取路径成本

        """
        return self.path_cost

    @path_cost.setter
    def path_cost(self, value: float):
        """
        设置路径成本

        """
        self.path_cost = value

    @property
    def laden_path_cost(self):
        return self.path_cost

    @property
    def empty_path_cost(self):
        return 0.5 * self.path_cost


    def add_transshipment(self, port: str, time: int):
        """
        添加转运港口和时间

        Args:
            port: 转运港口
            time: 转运时间
        """
        self.transshipment_port.append(port)
        self.transshipment_time.append(time)

    def add_port_in_path(self, port: Port):
        """
        添加路径中的港口

        Args:
            port: 要添加的港口对象
        """
        self.port_path.append(port.port)
        self.ports_in_path.append(port)
        self.number_of_path += 1

    def add_arc(self, arc: Arc):
        """
        添加弧

        Args:
            arc: 要添加的弧对象
        """
        self.arcs.append(arc)
        self.arcs_id.append(arc.arc_id)
        self.number_of_arcs += 1

    def __str__(self):
        """
        返回ContainerPath对象的字符串表示
        """
        return f"ContainerPath(id={self.container_path_id}, from={self.origin_port}, to={self.destination_port})"
