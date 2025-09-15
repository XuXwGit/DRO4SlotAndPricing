"""
@Author: XuXw
@Description: 集装箱路径类
@DateTime: 2024/12/4 21:54
"""
import logging
import os
from typing import List, Optional
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
    对应Java类: multi.network.ContainerPath
    
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
        self._container_path_id = container_path_id  # 对应Java: private int containerPathID
        self._origin_port = origin_port  # 对应Java: private String originPort
        self._origin_time = origin_time  # 对应Java: private int originTime
        self._destination_port = destination_port  # 对应Java: private String destinationPort
        self._destination_time = destination_time  # 对应Java: private int destinationTime
        self._path_time = destination_time - origin_time  # 对应Java: private int pathTime
        
        # 转运信息
        self._transshipment_port: List[str] = []  # 对应Java: private String[] transshipmentPort
        self._transshipment_time: List[int] = []  # 对应Java: private int[] transshipmentTime
        self._total_transship_time = 0  # 对应Java: private int totalTransshipTime
        self._transshipment_ports: List[Port] = []  # 对应Java: private List<Port> transshipmentPorts
        
        # 路径信息
        self._number_of_path = 0  # 对应Java: private int numberOfPath
        self._port_path: List[str] = []  # 对应Java: private String[] portPath
        self._ports_in_path: List[Port] = []  # 对应Java: private List<Port> portsInPath
        
        # 弧信息
        self._number_of_arcs = 0  # 对应Java: private int numberOfArcs
        self._arcs_id: List[int] = []  # 对应Java: private int[] arcsID
        self._arcs: List[Arc] = []  # 对应Java: private List<Arc> arcs
        
        # 成本
        self._path_cost = 0.0  # 对应Java: private double pathCost
    
    def get_total_transshipment_time(self) -> int:
        """
        计算总转运时间
        对应Java: public int getTotalTransshipmentTime()
        
        Returns:
            int: 总转运时间
        """
        # 对应Java: int totalTransshipmentTime = 0;
        total_transshipment_time = 0
        
        # 对应Java: if(transshipmentPort == null || transshipmentPort.length == 0)
        if not self._transshipment_port:
            # 对应Java: setTotalTransshipTime(0);
            self._total_transship_time = 0
            return 0
        
        # 对应Java: for (int i = 0; i < transshipmentPort.length; i++)
        for i in range(len(self._transshipment_port)):
            # 对应Java: totalTransshipmentTime += transshipmentTime[i];
            total_transshipment_time += self._transshipment_time[i]
        
        # 对应Java: setTotalTransshipTime(totalTransshipmentTime);
        self._total_transship_time = total_transshipment_time
        
        # 对应Java: return totalTransshipmentTime;
        return total_transshipment_time
    
    def get_total_demurrage_time(self) -> int:
        """
        计算总滞期时间
        对应Java: public int getTotalDemurrageTime()
        
        Returns:
            int: 总滞期时间
        """
        # 对应Java: int totalTransshipmentTime = 0;
        total_transshipment_time = 0
        # 对应Java: int totalDemurrageTime = 0;
        total_demurrage_time = 0
        
        # 对应Java: if(transshipmentPort == null || transshipmentPort.length == 0)
        if not self._transshipment_port:
            # 对应Java: setTotalTransshipTime(0);
            self._total_transship_time = 0
            # 对应Java: return 0;
            return 0
        
        # 对应Java: if(transshipmentPort.length != transshipmentTime.length)
        if len(self._transshipment_port) != len(self._transshipment_time):
            # 对应Java: log.info("Error in transshipment port num!");
            logger.info("Error in transshipment port num!")
            return 0
        
        # 对应Java: for (int i = 0; i < transshipmentPort.length; i++)
        for i in range(len(self._transshipment_port)):
            # 对应Java: totalTransshipmentTime += transshipmentTime[i];
            total_transshipment_time += self._transshipment_time[i]
            # 对应Java: if (transshipmentTime[i] > 7)
            if self._transshipment_time[i] > 7:
                # 对应Java: totalDemurrageTime += (transshipmentTime[i] - 7);
                total_demurrage_time += (self._transshipment_time[i] - 7)
        
        # 对应Java: setTotalTransshipTime(totalTransshipmentTime);
        self._total_transship_time = total_transshipment_time
        
        # 对应Java: return totalDemurrageTime;
        return total_demurrage_time
    

    @property
    def id(self) -> int:
        """
        获取路径ID
        对应Java: getId()
        """
        return self._container_path_id

    @property
    def path_id(self) -> int:
        """
        获取路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self._container_path_id

    @property
    def container_path_id(self) -> int:
        """
        获取集装箱路径ID（主ID，建议主流程全部用此属性）
        """
        return self._container_path_id

    @property
    def laden_path_id(self) -> int:
        """
        获取重箱路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self._container_path_id

    @property
    def empty_path_id(self) -> int:
        """
        获取空箱路径ID（别名，等价于container_path_id，建议主流程不用此属性）
        """
        return self._container_path_id


    @container_path_id.setter
    def container_path_id(self, value: int):
        """
        设置集装箱路径ID
        对应Java: setContainerPathID(int containerPathID)
        """
        self._container_path_id = value
    
    @property
    def origin_port(self) -> str:
        """
        获取起始港口
        对应Java: getOriginPort()
        """
        return self._origin_port
    
    @origin_port.setter
    def origin_port(self, value: str):
        """
        设置起始港口
        对应Java: setOriginPort(String originPort)
        """
        self._origin_port = value
    
    @property
    def origin_time(self) -> int:
        """
        获取起始时间
        对应Java: getOriginTime()
        """
        return self._origin_time
    
    @origin_time.setter
    def origin_time(self, value: int):
        """
        设置起始时间
        对应Java: setOriginTime(int originTime)
        """
        self._origin_time = value
    
    @property
    def destination_port(self) -> str:
        """
        获取目标港口
        对应Java: getDestinationPort()
        """
        return self._destination_port
    
    @destination_port.setter
    def destination_port(self, value: str):
        """
        设置目标港口
        对应Java: setDestinationPort(String destinationPort)
        """
        self._destination_port = value
    
    @property
    def destination_time(self) -> int:
        """
        获取目标时间
        对应Java: getDestinationTime()
        """
        return self._destination_time
    
    @destination_time.setter
    def destination_time(self, value: int):
        """
        设置目标时间
        对应Java: setDestinationTime(int destinationTime)
        """
        self._destination_time = value
    
    @property
    def travel_time(self) -> int:
        """
        获取运输时间
        对应Java: getTravelTime()
        """
        return self._path_time


    @property
    def path_time(self) -> int:
        """
        获取路径时间
        对应Java: getPathTime()
        """
        return self._path_time
    
    @path_time.setter
    def path_time(self, value: int):
        """
        设置路径时间
        对应Java: setPathTime(int pathTime)
        """
        self._path_time = value
    
    @property
    def travel_time_on_path(self) -> int:
        """
        获取路径上的运输时间
        对应Java: getTravelTimeOnPath()
        """
        return self._path_time


    @property
    def transshipment_port(self) -> List[str]:
        """
        获取转运港口列表
        对应Java: getTransshipmentPort()
        """
        return self._transshipment_port
    
    @transshipment_port.setter
    def transshipment_port(self, value: List[str]):
        """
        设置转运港口列表
        对应Java: setTransshipmentPort(String[] transshipmentPort)
        """
        self._transshipment_port = value
    
    @property
    def transshipment_time(self) -> List[int]:
        """
        获取转运时间列表
        对应Java: getTransshipmentTime()
        """
        return self._transshipment_time
    
    @transshipment_time.setter
    def transshipment_time(self, value: List[int]):
        """
        设置转运时间列表
        对应Java: setTransshipmentTime(int[] transshipmentTime)
        """
        self._transshipment_time = value
    
    @property
    def total_transship_time(self) -> int:
        """
        获取总转运时间
        对应Java: getTotalTransshipTime()
        """
        return self._total_transship_time
    
    @total_transship_time.setter
    def total_transship_time(self, value: int):
        """
        设置总转运时间
        对应Java: setTotalTransshipTime(int totalTransshipTime)
        """
        self._total_transship_time = value
    
    @property
    def transshipment_ports(self) -> List[Port]:
        """
        获取转运港口对象列表
        对应Java: getTransshipmentPorts()
        """
        return self._transshipment_ports
    
    @transshipment_ports.setter
    def transshipment_ports(self, value: List[Port]):
        """
        设置转运港口对象列表
        对应Java: setTransshipmentPorts(List<Port> transshipmentPorts)
        """
        self._transshipment_ports = value
    
    @property
    def number_of_path(self) -> int:
        """
        获取路径数量
        对应Java: getNumberOfPath()
        """
        return self._number_of_path
    
    @number_of_path.setter
    def number_of_path(self, value: int):
        """
        设置路径数量
        对应Java: setNumberOfPath(int numberOfPath)
        """
        self._number_of_path = value
    
    @property
    def port_path(self) -> List[str]:
        """
        获取港口路径
        对应Java: getPortPath()
        """
        return self._port_path
    
    @port_path.setter
    def port_path(self, value: List[str]):
        """
        设置港口路径
        对应Java: setPortPath(String[] portPath)
        """
        self._port_path = value
    
    @property
    def ports_in_path(self) -> List[Port]:
        """
        获取路径中的港口对象列表
        对应Java: getPortsInPath()
        """
        return self._ports_in_path
    
    @ports_in_path.setter
    def ports_in_path(self, value: List[Port]):
        """
        设置路径中的港口对象列表
        对应Java: setPortsInPath(List<Port> portsInPath)
        """
        self._ports_in_path = value
    
    @property
    def number_of_arcs(self) -> int:
        """
        获取弧数量
        对应Java: getNumberOfArcs()
        """
        return self._number_of_arcs
    
    @number_of_arcs.setter
    def number_of_arcs(self, value: int):
        """
        设置弧数量
        对应Java: setNumberOfArcs(int numberOfArcs)
        """
        self._number_of_arcs = value
    
    @property
    def arcs_id(self) -> List[int]:
        """
        获取弧ID列表
        对应Java: getArcsID()
        """
        return self._arcs_id
    
    @arcs_id.setter
    def arcs_id(self, value: List[int]):
        """
        设置弧ID列表
        对应Java: setArcsID(int[] arcsID)
        """
        self._arcs_id = value
    
    @property
    def arcs(self) -> List[Arc]:
        """
        获取弧对象列表
        对应Java: getArcs()
        """
        return self._arcs
    
    @arcs.setter
    def arcs(self, value: List[Arc]):
        """
        设置弧对象列表
        对应Java: setArcs(List<Arc> arcs)
        """
        self._arcs = value
    
    @property
    def cost(self) -> float:
        """
        获取路径成本
        对应Java: getCost()
        """
        return self._path_cost


    @property
    def path_cost(self) -> float:
        """
        获取路径成本
        对应Java: getPathCost()
        """
        return self._path_cost
    
    @path_cost.setter
    def path_cost(self, value: float):
        """
        设置路径成本
        对应Java: setPathCost(double pathCost)
        """
        self._path_cost = value
    
    @property
    def laden_path_cost(self):
        return self._path_cost
    
    @property
    def empty_path_cost(self):
        return 0.5 * self._path_cost


    def add_transshipment(self, port: str, time: int):
        """
        添加转运港口和时间
        
        Args:
            port: 转运港口
            time: 转运时间
        """
        self._transshipment_port.append(port)
        self._transshipment_time.append(time)
    
    def add_port_in_path(self, port: Port):
        """
        添加路径中的港口
        
        Args:
            port: 要添加的港口对象
        """
        self._port_path.append(port.port)
        self._ports_in_path.append(port)
        self._number_of_path += 1
    
    def add_arc(self, arc: Arc):
        """
        添加弧
        
        Args:
            arc: 要添加的弧对象
        """
        self._arcs.append(arc)
        self._arcs_id.append(arc.arc_id)
        self._number_of_arcs += 1
        
    def __str__(self):
        """
        返回ContainerPath对象的字符串表示
        """
        return f"ContainerPath(id={self.container_path_id}, from={self.origin_port}, to={self.destination_port})" 