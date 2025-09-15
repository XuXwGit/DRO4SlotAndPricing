"""
@Author: XuXw
@Description: 船舶路径类，对应Java版本VesselPath.java
@DateTime: 2024/12/4 21:54
"""
from typing import List
from ..network.arc import Arc


class VesselPath:
    """
    船舶路径类
    对应Java类: multi.network.VesselPath
    
    存储船舶路径相关的属性和信息
    """
    
    def __init__(self, vessel_path_id: int = 0, route_id: int = 0, number_of_arcs: int = 0, 
                 origin_time: int = 0, destination_time: int = 0, path_time: int = 0):
        """
        初始化船舶路径对象
        
        Args:
            vessel_path_id: 船舶路径ID，默认为0
            route_id: 航线ID，默认为0
            number_of_arcs: 弧的数量，默认为0
            origin_time: 起始时间，默认为0
            destination_time: 终止时间，默认为0
            path_time: 路径时间，默认为0
        """
        # 基本属性
        self._vessel_path_id = vessel_path_id  # 对应Java: private int vesselPathID
        self._route_id = route_id  # 对应Java: private int routeID
        
        # 弧信息
        self._number_of_arcs = number_of_arcs  # 对应Java: private int numberOfArcs
        self._arc_ids: List[int] = []  # 对应Java: private int[] arcIDs
        self._arcs: List[Arc] = []  # 对应Java: private List<Arc> arcs
        
        # 时间信息
        self._origin_time = origin_time  # 对应Java: private int originTime
        self._destination_time = destination_time  # 对应Java: private int destinationTime
        self._path_time = path_time  # 对应Java: private int pathTime
    
    # Getter和Setter方法
    @property
    def id(self) -> int:
        return self._vessel_path_id


    @property
    def vessel_path_id(self) -> int:
        """
        获取船舶路径ID
        对应Java: getVesselPathID()
        """
        return self._vessel_path_id
    
    @vessel_path_id.setter
    def vessel_path_id(self, value: int):
        """
        设置船舶路径ID
        对应Java: setVesselPathID(int vesselPathID)
        """
        self._vessel_path_id = value
    
    @property
    def route_id(self) -> int:
        """
        获取航线ID
        对应Java: getRouteID()
        """
        return self._route_id
    
    @route_id.setter
    def route_id(self, value: int):
        """
        设置航线ID
        对应Java: setRouteID(int routeID)
        """
        self._route_id = value
    
    @property
    def number_of_arcs(self) -> int:
        """
        获取弧的数量
        对应Java: getNumberOfArcs()
        """
        return self._number_of_arcs
    
    @number_of_arcs.setter
    def number_of_arcs(self, value: int):
        """
        设置弧的数量
        对应Java: setNumberOfArcs(int numberOfArcs)
        """
        self._number_of_arcs = value
    
    @property
    def arc_ids(self) -> List[int]:
        """
        获取弧ID列表
        对应Java: getArcIDs()
        """
        return self._arc_ids
    
    @arc_ids.setter
    def arc_ids(self, value: List[int]):
        """
        设置弧ID列表
        对应Java: setArcIDs(int[] arcIDs)
        """
        self._arc_ids = value
    
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
    def destination_time(self) -> int:
        """
        获取终止时间
        对应Java: getDestinationTime()
        """
        return self._destination_time
    
    @destination_time.setter
    def destination_time(self, value: int):
        """
        设置终止时间
        对应Java: setDestinationTime(int destinationTime)
        """
        self._destination_time = value
    
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
        
    def __str__(self) -> str:
        """
        返回船舶路径的字符串表示
        
        Returns:
            str: 船舶路径的字符串描述
        """
        return f"VesselPath(id={self.vessel_path_id}, route_id={self.route_id}, path_time={self.path_time})" 