from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .port import Port
from .vessel_path import VesselPath
from .vessel_type import VesselType

@dataclass
class ShipRoute:
    """
    航线类
    
    存储航线相关的属性和信息
    
    属性:
        ship_route_id: 航线ID
        cycle_time: 周期时间
        num_round_trips: 往返次数
        number_of_ports: 港口数量
        ports: 港口列表
        port_calls: 港口停靠映射(停靠索引->港口)
        number_of_call: 停靠次数
        ports_of_call: 停靠港口列表
        time_points_of_call: 停靠时间点列表
        num_vessel_paths: 船舶路径数量
        vessel_paths: 船舶路径列表
        vessel_type: 船舶类型
        fleet: 船队映射(轮次索引->船舶类型)
        available_vessels: 可用船舶映射(船舶ID->船舶类型)
    """
    def __init__(self, 
                 ship_route_id: int, 
                 cycle_time: int, 
                 number_of_ports: int,      
                 number_of_call: int,
                 ports_of_call: List[str], 
                 time_points_of_call: List[int],
                 ):
        """对应Java: public ShipRoute()"""
        # 基本属性
        self._ship_route_id: int = ship_route_id  # 对应Java: private int shipRouteID
        self._cycle_time: int = cycle_time  # 对应Java: private int cycleTime
        
        # 往返和港口信息
        self._num_round_trips: int = 0  # 对应Java: private int numRoundTrips
        self._number_of_ports: int = number_of_ports  # 对应Java: private int numberOfPorts
        self._ports: List[str] = ports_of_call  # 对应Java: private String[] ports
        
        # 港口挂靠信息
        # key: port call index, value: Port
        self._port_calls: Dict[int, Port] = {}  # 对应Java: private Map<Integer, Port> portCalls
        self._number_of_call: int = number_of_call  # 对应Java: private int numberOfCall
        self._ports_of_call: List[str] = ports_of_call  # 对应Java: private String[] portsOfCall
        self._time_points_of_call: List[int] = time_points_of_call  # 对应Java: private int[] timePointsOfCall
        
        # 船舶路径信息
        # key: Rotation index in the planning horizon, value: VesselPath
        self._num_vessel_paths: int = 0  # 对应Java: private int numVesselPaths
        self._vessel_paths: List[VesselPath] = []  # 对应Java: private List<VesselPath> vesselPaths
        
        # 船型分配
        self._vessel_type: Optional[VesselType] = None  # 对应Java: private VesselType vesselType
        
        # key: Rotation index in the planning horizon, value: VesselType Object
        self._fleet: Dict[int, VesselType] = {}  # 对应Java: private Map<Integer, VesselType> fleet
        
        # key: vesselID, value: VesselType Object
        self._available_vessels: Dict[int, VesselType] = {}  # 对应Java: private Map<Integer, VesselType> availableVessels
    
    def get_call_index_of_port(self, port: str) -> int:
        """
        获取港口的挂靠索引
        对应Java: public int getCallIndexOfPort(String port)
        """
        # 对应Java: for (int p = 0; p < this.numberOfCall - 1; p++)
        for p in range(self._number_of_call - 1):
            # 对应Java: if(port.equals(this.getPortsOfCall()[p]))
            if port == self._ports_of_call[p]:
                # 对应Java: return p;
                return p
        # 对应Java: return -1;
        return -1
    
    # Getter和Setter方法
    @property
    def id(self) -> int:
        """对应Java: getId()"""
        return self._ship_route_id

    @id.setter
    def id(self, value: int):
        """对应Java: setId(int id)"""
        self._ship_route_id = value

    @property
    def ship_route_id(self) -> int:
        """对应Java: getShipRouteID()"""
        return self._ship_route_id
    
    @ship_route_id.setter
    def ship_route_id(self, value: int):
        """对应Java: setShipRouteID(int shipRouteID)"""
        self._ship_route_id = value
    
    @property
    def route_id(self) -> int:
        """对应Java: getRouteID()"""
        return self._ship_route_id
    
    @route_id.setter
    def route_id(self, value: int):
        """对应Java: setRouteID(int routeID)"""
        self._ship_route_id = value

    @property
    def cycle_time(self) -> int:
        """对应Java: getCycleTime()"""
        return self._cycle_time
    
    @cycle_time.setter
    def cycle_time(self, value: int):
        """对应Java: setCycleTime(int cycleTime)"""
        self._cycle_time = value
    
    @property
    def num_round_trips(self) -> int:
        """对应Java: getNumRoundTrips()"""
        return self._num_round_trips
    
    @num_round_trips.setter
    def num_round_trips(self, value: int):
        """对应Java: setNumRoundTrips(int numRoundTrips)"""
        self._num_round_trips = value
    
    @property
    def number_of_ports(self) -> int:
        """对应Java: getNumberOfPorts()"""
        return self._number_of_ports
    
    @number_of_ports.setter
    def number_of_ports(self, value: int):
        """对应Java: setNumberOfPorts(int numberOfPorts)"""
        self._number_of_ports = value
    
    @property
    def ports(self) -> List[str]:
        """对应Java: getPorts()"""
        return self._ports
    
    @ports.setter
    def ports(self, value: List[str]):
        """对应Java: setPorts(String[] ports)"""
        self._ports = value
    
    @property
    def port_calls(self) -> Dict[int, Port]:
        """对应Java: getPortCalls()"""
        return self._port_calls
    
    @port_calls.setter
    def port_calls(self, value: Dict[int, Port]):
        """对应Java: setPortCalls(Map<Integer, Port> portCalls)"""
        self._port_calls = value
    
    @property
    def number_of_call(self) -> int:
        """对应Java: getNumberOfCall()"""
        return self._number_of_call
    
    @number_of_call.setter
    def number_of_call(self, value: int):
        """对应Java: setNumberOfCall(int numberOfCall)"""
        self._number_of_call = value
    
    @property
    def ports_of_call(self) -> List[str]:
        """对应Java: getPortsOfCall()"""
        return self._ports_of_call
    
    @ports_of_call.setter
    def ports_of_call(self, value: List[str]):
        """对应Java: setPortsOfCall(String[] portsOfCall)"""
        self._ports_of_call = value
    
    @property
    def time_points_of_call(self) -> List[int]:
        """对应Java: getTimePointsOfCall()"""
        return self._time_points_of_call
    
    @time_points_of_call.setter
    def time_points_of_call(self, value: List[int]):
        """对应Java: setTimePointsOfCall(int[] timePointsOfCall)"""
        self._time_points_of_call = value
    
    @property
    def num_vessel_paths(self) -> int:
        """对应Java: getNumVesselPaths()"""
        return self._num_vessel_paths
    
    @num_vessel_paths.setter
    def num_vessel_paths(self, value: int):
        """对应Java: setNumVesselPaths(int numVesselPaths)"""
        self._num_vessel_paths = value
    
    @property
    def vessel_paths(self) -> List[VesselPath]:
        """对应Java: getVesselPaths()"""
        return self._vessel_paths
    
    @vessel_paths.setter
    def vessel_paths(self, value: List[VesselPath]):
        """对应Java: setVesselPaths(List<VesselPath> vesselPaths)"""
        self._vessel_paths = value
    
    @property
    def vessel_type(self) -> Optional[VesselType]:
        """对应Java: getVesselType()"""
        return self._vessel_type
    
    @vessel_type.setter
    def vessel_type(self, value: Optional[VesselType]):
        """对应Java: setVesselType(VesselType vesselType)"""
        self._vessel_type = value
    
    @property
    def fleet(self) -> Dict[int, VesselType]:
        """对应Java: getFleet()"""
        return self._fleet
    
    @fleet.setter
    def fleet(self, value: Dict[int, VesselType]):
        """对应Java: setFleet(Map<Integer, VesselType> fleet)"""
        self._fleet = value
    
    @property
    def available_vessels(self) -> Dict[int, VesselType]:
        """对应Java: getAvailableVessels()"""
        return self._available_vessels
    
    @available_vessels.setter
    def available_vessels(self, value: Dict[int, VesselType]):
        """对应Java: setAvailableVessels(Map<Integer, VesselType> availableVessels)"""
        self._available_vessels = value

    def add_port_call(self, port: Port):
        """
        添加港口停靠
        
        Args:
            port: 港口对象 

        Returns:
            int: 停靠索引，如果未找到则返回-1
        """
        self._ports_of_call.append(port)
        self._number_of_call += 1
        return self._number_of_call - 1 