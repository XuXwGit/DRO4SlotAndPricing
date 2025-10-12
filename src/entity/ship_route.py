from typing import List, Dict, Optional
from .port import Port
from .vessel_path import VesselPath
from .vessel_type import VesselType


class ShipRoute:
    """
    航线类
    
    存储航线相关的属性和信息
    """

    def __init__(
        self,
        ship_route_id: int,
        cycle_time: int,
        number_of_ports: int,
        number_of_call: int,
        ports_of_call: List[str],          # 保留用于显示/日志
        time_points_of_call: List[int],
        port_objects: List[Port],          # ← 新增：实际 Port 对象列表
    ):
        self._ship_route_id = ship_route_id
        self._cycle_time = cycle_time
        self._number_of_ports = number_of_ports
        self._number_of_call = number_of_call
        self._ports_of_call = ports_of_call.copy()
        self._time_points_of_call = time_points_of_call.copy()
        
        # 关键：使用传入的 Port 对象
        self._ports = port_objects
        self._port_calls = {i: port for i, port in enumerate(port_objects)}
    
        # 将港口名称转换为 Port 对象，并构建 port_calls 映射
        self._ports: List[Port] = [Port() for name in ports_of_call]
        self._port_calls: Dict[int, Port] = {
            i: Port() for i, name in enumerate(ports_of_call)
        }

        # 船舶路径信息
        self._num_vessel_paths: int = 0
        self._vessel_paths: List[VesselPath] = []

        # 船型分配
        self._vessel_type: Optional[VesselType] = None
        self._fleet: Dict[int, VesselType] = {}
        self._available_vessels: Dict[int, VesselType] = {}

    def get_call_index_of_port(self, port_name: str) -> int:
        """
        获取港口名称在 ports_of_call 中第一次出现的挂靠索引。

        Args:
            port_name (str): 港口名称

        Returns:
            int: 第一次出现的索引，若未找到则返回 -1
        """
        try:
            return self._ports_of_call.index(port_name)
        except ValueError:
            return -1

    def get_all_call_indices_of_port(self, port_name: str) -> List[int]:
        """
        获取港口名称在 ports_of_call 中所有出现的挂靠索引。

        Args:
            port_name (str): 港口名称

        Returns:
            List[int]: 所有匹配的索引列表
        """
        return [i for i, name in enumerate(self._ports_of_call) if name == port_name]

    # ========== 属性访问器 ==========

    @property
    def id(self) -> int:
        """航线ID（别名）"""
        return self._ship_route_id

    @id.setter
    def id(self, value: int):
        self._ship_route_id = value

    @property
    def ship_route_id(self) -> int:
        """航线ID"""
        return self._ship_route_id

    @ship_route_id.setter
    def ship_route_id(self, value: int):
        self._ship_route_id = value

    @property
    def route_id(self) -> int:
        """航线ID（别名）"""
        return self._ship_route_id

    @route_id.setter
    def route_id(self, value: int):
        self._ship_route_id = value

    @property
    def cycle_time(self) -> int:
        """周期时间（单位：小时或天，依上下文而定）"""
        return self._cycle_time

    @cycle_time.setter
    def cycle_time(self, value: int):
        self._cycle_time = value

    @property
    def num_round_trips(self) -> int:
        """往返次数"""
        return self._num_round_trips

    @num_round_trips.setter
    def num_round_trips(self, value: int):
        self._num_round_trips = value

    @property
    def number_of_ports(self) -> int:
        """港口数量（不重复？或指航线设计中的港口数）"""
        return self._number_of_ports

    @number_of_ports.setter
    def number_of_ports(self, value: int):
        self._number_of_ports = value

    @property
    def ports(self) -> List[Port]:
        """港口对象列表（由 ports_of_call 构建）"""
        return self._ports

    @ports.setter
    def ports(self, value: List[Port]):
        self._ports = value

    @property
    def port_calls(self) -> Dict[int, Port]:
        """挂靠索引到 Port 对象的映射"""
        return self._port_calls

    @port_calls.setter
    def port_calls(self, value: Dict[int, Port]):
        self._port_calls = value

    @property
    def number_of_call(self) -> int:
        """总挂靠次数（包括重复港口）"""
        return self._number_of_call

    @number_of_call.setter
    def number_of_call(self, value: int):
        self._number_of_call = value

    @property
    def ports_of_call(self) -> List[str]:
        """挂靠港口名称列表（按顺序，可重复）"""
        return self._ports_of_call

    @ports_of_call.setter
    def ports_of_call(self, value: List[str]):
        self._ports_of_call = value.copy()
        # 同步更新 ports 和 port_calls
        self._ports = [Port() for name in value]
        self._port_calls = {i: Port() for i, name in enumerate(value)}
        self._number_of_call = len(value)

    @property
    def time_points_of_call(self) -> List[int]:
        """各挂靠点的时间点（相对时间）"""
        return self._time_points_of_call

    @time_points_of_call.setter
    def time_points_of_call(self, value: List[int]):
        self._time_points_of_call = value.copy()

    @property
    def num_vessel_paths(self) -> int:
        """船舶路径数量"""
        return self._num_vessel_paths

    @num_vessel_paths.setter
    def num_vessel_paths(self, value: int):
        self._num_vessel_paths = value

    @property
    def vessel_paths(self) -> List[VesselPath]:
        """船舶路径列表"""
        return self._vessel_paths

    @vessel_paths.setter
    def vessel_paths(self, value: List[VesselPath]):
        self._vessel_paths = value
        self._num_vessel_paths = len(value)

    @property
    def vessel_type(self) -> Optional[VesselType]:
        """默认船舶类型"""
        return self._vessel_type

    @vessel_type.setter
    def vessel_type(self, value: Optional[VesselType]):
        self._vessel_type = value

    @property
    def fleet(self) -> Dict[int, VesselType]:
        """轮次索引到船舶类型的映射"""
        return self._fleet

    @fleet.setter
    def fleet(self, value: Dict[int, VesselType]):
        self._fleet = value

    @property
    def available_vessels(self) -> Dict[int, VesselType]:
        """可用船舶映射：vesselID -> VesselType"""
        return self._available_vessels

    @available_vessels.setter
    def available_vessels(self, value: Dict[int, VesselType]):
        self._available_vessels = value


    def add_port_call(self, port_name: str, time_point: int) -> None:
        """
        添加一个新的港口挂靠记录。

        Args:
            port_name (str): 港口名称
            time_point (int): 挂靠时间点（相对于航线起点）
        """
        call_index = self._number_of_call  # 新挂靠的索引

        # 更新列表
        self._ports_of_call.append(port_name)
        self._time_points_of_call.append(time_point)

        # 创建 Port 对象并更新映射
        port_obj = Port()
        self._ports.append(port_obj)
        self._port_calls[call_index] = port_obj

        # 更新计数
        self._number_of_call += 1