import logging

import time
from typing import List, Dict, Any, Literal
import pandas as pd
import sqlite3

# 添加项目根目录到路径
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.config.config import Config
from src.config.data_config import DB_COLUMN_MAPS
from src.entity import ContainerPath, Port, ShipRoute, VesselType, VesselPath, ODRange, Request, LadenPath, EmptyPath
from src.network import Node, TravelingArc, TransshipArc
from src.utils.data_manager import DataManager

logger = logging.getLogger(__name__)

class DataReader:
    """
    数据读取类，用于从本地文件读取输入数据。

    该类负责读取所有必要的数据文件，包括：
    - 航运网络数据（港口和航线）
    - 时空网络数据（节点、航行弧、转运弧）
    - 集装箱路径数据（重箱路径和空箱路径）
    - 船舶数据
    - 订单/请求数据
    - 历史解决方案和需求样本场景

    Attributes:
        input_data: 输入数据对象
        time_horizon: 时间范围
        file_path: 数据文件路径
        db_path: 数据库文件路径
        use_db: 是否使用数据库读取数据
    """

    def __init__(self,
                 input_data: DataManager,
                 time_horizon: int,
                 use_db: bool = False,
                 local_path: str = Config.ROOT_PATH + "/" + Config.DATA_PATH,
                 db_path: str = ""):
        """
        初始化数据读取器

        Args:
            path: 数据文件相对路径
            input_data: 输入数据对象
            time_horizon: 时间范围
            use_db: 是否使用数据库读取数据
            db_path: 数据库文件路径
        """
        self.input_data = input_data
        self.time_horizon = time_horizon
        self.use_db = use_db
        self.file_path = local_path
        self.db_path = db_path


    def read(self, case: str = "1"):
        """
        数据读取的主框架，按顺序读取所有必要的数据
        """
        if case != "":
            self.file_path = self.file_path + case + "/"

        logger.info("========Start to read data========")
        start_time = time.time()

        self.input_data.time_horizon = self.time_horizon

        # 1. 读取航运网络数据（包括港口和航线）
        self._read_ports()
        self._read_ship_routes()

        # 2. 读取时空网络数据（包括节点、航行弧、转运弧）
        self._read_nodes()
        self._read_traveling_arcs()
        self._read_transship_arcs()

        # 3. 读取集装箱路径数据（包括重箱路径和空箱路径）
        self._read_container_paths()
        self._read_vessel_paths()
        # self._read_laden_paths()
        # self._read_empty_paths()

        # 4. 读取船舶数据
        self._read_vessels()

        # 5. 读取订单/请求数据
        self._read_demand_range()
        self._read_requests()

        # 6. 读取历史解决方案和样本场景
        if Config.USE_HISTORY_SOLUTION:
            self._read_history_solution()
        if Config.WHETHER_LOAD_SAMPLE_TESTS:
            self._read_sample_scenes()

        if Config.WHETHER_PRINT_DATA_STATUS:
            self.input_data.show_status()

        end_time = time.time()
        logger.info(f"========End read data ({end_time - start_time:.2f}s)========")

        return self.input_data

    def _read_to_dataframe(self, filename: str) -> pd.DataFrame:
        """
        从本地文件或数据库读取数据到DataFrame

        Args:
            filename: 文件名或表名

        Returns:
            pd.DataFrame: 包含文件数据的DataFrame
        """
        try:
            if self.use_db and self.db_path:
                table_name = Config.FILE_TABLE_MAP.get(filename, None)
                if table_name is None:
                    raise ValueError(f"Table name not found for file: {filename}")
                logger.info(f"Reading from database table: {table_name}")
                conn = sqlite3.connect(self.db_path)
                df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                conn.close()
                # 统一字段名
                std_map = DB_COLUMN_MAPS.get(table_name, None)
                # logger.info(f"std_map: {std_map}")
                if std_map:
                    df.rename(columns=std_map, inplace=True)
                # logger.info(f"df: {df}")
                return df
            else:
                file_path = self.file_path + filename
                logger.info(f"Reading from file: {file_path}")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found: {file_path}")

                # 自动判断文件类型
                ext = os.path.splitext(file_path)[1].lower()

                if ext in ['.csv']:
                    df = pd.read_csv(file_path)
                elif ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(file_path)
                elif ext in ['.txt']:
                    df = DataReader._read_txt_smart(file_path)
                else:
                    raise ValueError(f"Unsupported file type: {ext}")
                return df
        except Exception as e:
            logger.error(f"Error reading data {filename}: {str(e)}")
            raise


    @staticmethod
    def _read_txt_smart(filepath: str, **read_kwargs) -> pd.DataFrame:
        """
        智能读取 .txt 文件：
        - 如果包含分隔符（如 , \t | ;），则按分隔符解析为表格
        - 否则尝试逐行读取为单列 DataFrame（如日志、关键词列表）
        - 自动检测是否含 header
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            # 读取前几行用于分析
            sample_lines = [first_line]
            for _ in range(3):  # 最多再读3行
                line = f.readline().strip()
                if line:
                    sample_lines.append(line)
                else:
                    break

        # 检查是否有明显分隔符
        separators = {
            ',': 'csv-like',
            '\t': 'tsv-like',
            '|': 'pipe-separated',
            ';': 'semicolon-separated'
        }

        detected_sep = None
        for sep, desc in separators.items():
            if sep in first_line and len(first_line.split(sep)) > 1:
                detected_sep = sep
                break

        # 情况1：检测到分隔符 → 当作表格处理
        if detected_sep:
            # 尝试读取为表格，自动推断 header
            df = pd.read_csv(
                filepath,
                sep=detected_sep,
                engine='python',  # 更灵活的解析器
                **read_kwargs
            )
            print(f"✅ Detected {detected_sep}-separated table in {filepath}")
            return df

        # 情况2：无分隔符，但有多行 → 视为单列数据（如关键词、ID 列表）
        elif len(sample_lines) > 1:
            df = pd.DataFrame({os.path.basename(filepath).replace('.txt', ''): sample_lines})
            print(f"✅ Detected multi-line text (single column) from {filepath}")
            return df

        # 情况3：单行文本 → 作为单值返回（罕见，但仍支持）
        else:
            content = first_line
            df = pd.DataFrame({'content': [content]})
            print(f"⚠️ Detected single-line text in {filepath}")
            return df



    def _read_demand_range(self):
        """
        读取需求范围数据

        文件格式：
        OriginRegion DestinationRegion DemandLowerBound DemandUpperBound FreightLowerBound FreightUpperBound
        1 2 20 40 2000 2500
        2 1 40 60 2000 2500
        ...
        """
        df = self._read_to_dataframe(Config.DEMAND_RANGE_FILENAME)
        self.input_data.data['demand'] = df

        range_map = {}

        for _, row in df.iterrows():
            od_range = ODRange(
                origin_group=int(row["OriginRegion"]),
                destination_group=int(row["DestinationRegion"]),
                demand_lower_bound=int(row["DemandLowerBound"]),
                demand_upper_bound=int(row["DemandUpperBound"]),
                freight_lower_bound=int(row["FreightLowerBound"]),
                freight_upper_bound=int(row["FreightUpperBound"])
            )
            key = f"{row['OriginRegion']}{row['DestinationRegion']}"
            range_map[key] = od_range

        self.input_data.group_range_map = range_map

    def _read_container_paths(self):
        """
        读取集装箱路径数据

        文件格式：
        ContainerPathID OriginPort OriginTime DestinationPort DestinationTime PathTime TransshipPort TransshipTime PortPath_length PortPath Arcs_length Arcs_ID
        1 A 2 B 3 1 0 0 2 A,B 1 1
        2 A 9 B 10 1 0 0 2 A,B 1 9
        ...
        """
        df = self._read_to_dataframe(Config.PATHS_FILENAME)
        self.input_data.data['container_path'] = df

        container_path_list = []
        container_paths = {}

        for _, row in df.iterrows():
            if int(row['DestinationTime']) > self.time_horizon:
                continue

            container_path = ContainerPath(
                container_path_id=int(row["ContainerPathID"]),
                origin_port=row["OriginPort"],
                origin_time=int(row["OriginTime"]),
                destination_port=row["DestinationPort"],
                destination_time=int(row["DestinationTime"]),
            )

            # 设置转运信息
            if row['TransshipPort'] != '0':  # 如果有转运港口
                transship_ports = row['TransshipPort'].split(',')
                transship_times = [int(t) for t in row['TransshipTime'].split(',')]
                for port, time in zip(transship_ports, transship_times):
                    container_path.add_transshipment(
                        self.input_data.port_set[port].port,
                        time
                    )

            # 设置港口路径
            port_path = row['PortPath'].split(',')
            for port in port_path:
                container_path.add_port_in_path(self.input_data.port_set[port])

            # 设置弧
            arc_ids = [int(aid) for aid in row['ArcsID'].split(',')]
            for arc_id in arc_ids:
                container_path.add_arc(self.input_data.arc_set[arc_id])

            container_path_list.append(container_path)
            container_paths[container_path.container_path_id] = container_path

        self.input_data.container_paths = container_path_list
        self.input_data.container_path_set = container_paths

    def _read_ports(self):
        """
        读取港口数据
        文件格式：PortID Port WhetherTrans Region Group
        """
        df = self._read_to_dataframe(Config.PORTS_FILENAME)
        self.input_data.data['ports'] = df
        port_list = []
        ports = {}

        # 设置默认值
        default_turn_over_time = Config.DEFAULT_TURN_OVER_TIME
        default_laden_demurrage_cost = Config.DEFAULT_LADEN_DEMURRAGE_COST
        default_empty_demurrage_cost = Config.DEFAULT_EMPTY_DEMURRAGE_COST
        default_unit_loading_cost = Config.DEFAULT_UNIT_LOADING_COST
        default_unit_discharge_cost = Config.DEFAULT_UNIT_DISCHARGE_COST
        default_unit_transshipment_cost = Config.DEFAULT_UNIT_TRANSSHIPMENT_COST
        default_unit_rental_cost = Config.DEFAULT_UNIT_RENTAL_COST

        for _, row in df.iterrows():
            port = Port(
                id=int(row["PortID"]),
                port=row["Port"],
                whether_trans=int(row["WhetherTrans"]),
                region=row["Region"],
                group=int(row["Group"]),
                turn_over_time=Config.DEFAULT_TURN_OVER_TIME,
                laden_demurrage_cost=Config.DEFAULT_LADEN_DEMURRAGE_COST,
                empty_demurrage_cost=Config.DEFAULT_EMPTY_DEMURRAGE_COST,
                loading_cost=Config.DEFAULT_UNIT_LOADING_COST,
                discharge_cost=Config.DEFAULT_UNIT_DISCHARGE_COST,
                transshipment_cost=Config.DEFAULT_UNIT_TRANSSHIPMENT_COST,
                rental_cost=Config.DEFAULT_UNIT_RENTAL_COST
            )
            port_list.append(port)
            ports[port.port] = port
        logger.info(f"success to load ports")
        self.input_data.ports = port_list
        self.input_data.port_set = ports

    def _read_ship_routes(self):
        """
        读取航线数据

        文件格式：
        ID CycleTime NumRoundTrips NumberOfPorts PortCallSequence
        1 10 2 3 A,B,C
        2 15 1 4 A,C,D,B
        ...
        """
        df = self._read_to_dataframe(Config.ROUTES_FILENAME)
        self.input_data.data['ship_routes'] = df

        route_list = []
        routes = {}

        for _, row in df.iterrows():
            port_names = row["PortsOfCall"].split(',')
            time_points = [int(t) for t in row["Time"].split(',')]

            # 从 input_data 获取 Port 对象
            port_objects = [self.input_data.port_set[name] for name in port_names]

            route = ShipRoute(
                ship_route_id=int(row["ShippingRouteID"]),
                cycle_time=time_points[-1] - time_points[0],
                number_of_ports=int(row["NumberOfPorts"]),
                number_of_call=len(port_names),
                ports_of_call=port_names,
                time_points_of_call=time_points,
                port_objects=port_objects,  # ← 传入 Port 对象
            )

            route_list.append(route)
            routes[route.ship_route_id] = route

        self.input_data.ship_routes = route_list
        self.input_data.ship_route_set = routes

    def _read_nodes(self):
        """
        读取节点数据
        文件格式：ID Route Call Port RoundTrip Time
        """
        df = self._read_to_dataframe(Config.NODES_FILENAME)
        self.input_data.data['nodes'] = df

        node_list = []
        nodes = {}

        for _, row in df.iterrows():
            if int(row["Time"]) > self.time_horizon:
                continue

            node = Node(
                id=int(row["ID"]),
                route=int(row["Route"]),
                call=int(row["Call"]),
                port_string=row["Port"],
                round_trip=int(row["RoundTrip"]),
                time=int(row["Time"]),
                port=self.input_data.port_set[row["Port"]]
            )
            if node.time <= self.time_horizon:
                node_list.append(node)
                nodes[node.node_id] = node
        logger.info(f"success to load nodes")
        self.input_data.nodes = node_list
        self.input_data.node_set = nodes

    def _read_traveling_arcs(self):
        """
        读取航行弧数据
        文件格式：TravelingArcID Route OriginNodeID OriginCall OriginPort RoundTrip OriginTime TravelingTime DestinationNodeID DestinationCall DestinationPort DestinationTime
        """
        df = self._read_to_dataframe(Config.TRAVELING_ARCS_FILENAME)
        self.input_data.data['traveling_arcs'] = df

        arc_list = []
        arcs = {}

        for _, row in df.iterrows():
            if int(row["DestinationTime"]) > self.time_horizon:
                continue
            origin_node = self.input_data.node_set[int(row["OriginNodeID"])]
            destination_node = self.input_data.node_set[int(row["DestinationNodeID"])]
            arc = TravelingArc(
                arc_id=int(row["TravelingArcID"]),
                route_id=int(row["Route"]),
                round_trip=int(row["RoundTrip"]),
                travel_time=int(row["TravelingTime"]),
                origin_node=origin_node,
                destination_node=destination_node,
            )
            if arc.origin_node.time <= self.time_horizon and arc.destination_node.time <= self.time_horizon:
                arc_list.append(arc)
                arcs[arc.arc_id] = arc
        logger.info(f"success to load traveling arcs")
        self.input_data.traveling_arcs = arc_list
        self.input_data.traveling_arc_set = arcs

    def _read_transship_arcs(self):
        """
        读取转运弧数据
        文件格式：TransshipArcID Port OriginNodeID OriginTime TransshipTime DestinationNodeID DestinationTime FromRoute ToRoute
        """
        df = self._read_to_dataframe(Config.TRANSSHIP_ARCS_FILENAME)
        self.input_data.data['transship_arcs'] = df

        arc_list = []
        arcs = {}

        for _, row in df.iterrows():
            if int(row["DestinationTime"]) >self.time_horizon:
                continue
            port = self.input_data.port_set[row["Port"]]
            arc = TransshipArc(
                id=int(row["TransshipArcID"]),
                port=port.port,
                origin_node=self.input_data.node_set[int(row["OriginNodeID"])],
                destination_node=self.input_data.node_set[int(row["DestinationNodeID"])],
                from_route=int(row["FromRoute"]),
                to_route=int(row["ToRoute"]),
                transship_time=int(row["TransshipTime"]),
            )
            if arc.origin_node.time <= self.time_horizon and arc.destination_node.time <= self.time_horizon:
                arc_list.append(arc)
                arcs[arc.arc_id] = arc
        logger.info(f"success to load transship arcs")
        self.input_data.transship_arcs = arc_list
        self.input_data.transship_arc_set = arcs

        # 合并所有弧
        self.input_data.arcs = self.input_data.traveling_arcs + self.input_data.transship_arcs # type: ignore
        self.input_data.arc_set = {**self.input_data.traveling_arc_set, **self.input_data.transship_arc_set}

    def _read_vessel_paths(self):
        """
        读取船舶路径数据
        文件格式：VesselPathID ShippingRouteID NumberOfArcs ArcIDs OriginTime DestinationTime PathTime
        """
        df = self._read_to_dataframe("VesselPaths.txt")
        self.input_data.data['vessel_paths'] = df

        vessel_path_list = []
        vessel_paths = {}

        for _, row in df.iterrows():
            if int(row["DestinationTime"]) > self.time_horizon:
                continue
            vessel_path = VesselPath(
                vessel_path_id=int(row["VesselPathID"]),
                route_id=int(row["ShippingRouteID"]),
                number_of_arcs=int(row["NumberOfArcs"]),
                origin_time=int(row["OriginTime"]),
                destination_time=int(row["DestinationTime"]),
            )
            arc_ids = [int(aid) for aid in row["ArcIDs"].split(",")]
            vessel_path.arc_ids = arc_ids
            vessel_path.arcs = [self.input_data.arc_set[aid] for aid in arc_ids]

            if vessel_path.origin_time <= self.time_horizon and vessel_path.destination_time <= self.time_horizon:
                vessel_path_list.append(vessel_path)
                vessel_paths[vessel_path.vessel_path_id] = vessel_path
        logger.info(f"success to load vessel paths")
        self.input_data.vessel_paths = vessel_path_list
        self.input_data.vessel_path_set = vessel_paths

    def _read_laden_paths(self):
        """
        读取重箱路径数据
        文件格式：RequestID OriginPort OriginTime DestinationPort RoundTrip EarliestSetUpTime ArrivalTimeToDestination PathTime TransshipPort TransshipTime PathID PortPath ArcIDs
        """
        df = self._read_to_dataframe(Config.LADEN_PATHS_FILENAME)
        self.input_data.data['laden_paths'] = df

        laden_path_list = []
        laden_paths = {}

        for _, row in df.iterrows():
            if int(row["ArrivalTimeToDestination"]) > self.time_horizon:
                continue
            laden_path = LadenPath(
                request_id=int(row["RequestID"]),
                origin_port=row["OriginPort"],
                origin_time=int(row["OriginTime"]),
                destination_port=row["DestinationPort"],
                round_trip=int(row["RoundTrip"]),
                earliest_setup_time=int(row["EarliestSetUpTime"]),
                arrival_time_to_destination=int(row["ArrivalTimeToDestination"]),
                path_time=int(row["PathTime"]),
                ports_in_path = row["PortPath"].split(","),
                path_id=int(row["PathID"]),
                arc_ids=[int(aid) for aid in row["ArcIDs"].split(",")],
                arcs = [self.input_data.arc_set[int(aid)] for aid in row["ArcIDs"]]
            )

            if laden_path.origin_time <= self.time_horizon and laden_path.arrival_time_to_destination <= self.time_horizon:
                laden_path_list.append(laden_path)
                laden_paths[laden_path.path_id] = laden_path
        logger.info(f"success to load laden paths")
        self.input_data.laden_paths = laden_path_list
        self.input_data.laden_path_set = laden_paths

    def _read_empty_paths(self):
        """
        读取空箱路径数据
        文件格式：RequestID OriginPort OriginTime NumOfEmptyPath PathIDs
        """
        df = self._read_to_dataframe(Config.EMPTY_PATHS_FILENAME)
        self.input_data.data['empty_paths'] = df

        empty_path_list = []
        empty_paths = {}

        for _, row in df.iterrows():
            if int(row["NumOfEmptyPath"]) <= 0:
                continue

            if int(row["RequestID"]) not in self.input_data.request_set:
                continue

            for path_id_str in row["PathIDs"].split(","):
                empty_path = EmptyPath(
                    path_id=int(path_id_str),
                    request_id=int(row["RequestID"]),
                    origin_port_string=row["OriginPort"],
                    origin_time=int(row["OriginTime"]),
                    container_path=self.input_data.container_path_set.get(int(path_id_str)),
                    origin_port=self.input_data.port_set[row["OriginPort"]]
                )
                if empty_path.origin_time <= self.time_horizon:
                    empty_path_list.append(empty_path)
                    empty_paths[empty_path.path_id] = empty_path
        logger.info(f"success to load empty paths: {empty_paths}")
        self.input_data.empty_paths = empty_path_list
        self.input_data.empty_path_set = empty_paths

    def _read_requests(self):
        """
        读取请求数据
        文件格式：RequestID OriginPort DestinationPort EarliestPickupTime LatestDestinationTime LadenPaths NumberOfLadenPath EmptyPaths NumberOfEmptyPath
        """
        df = self._read_to_dataframe(Config.REQUESTS_FILENAME)
        self.input_data.data['requests'] = df

        request_list = []
        requests = {}
        try:
            for _, row in df.iterrows():
                if int(row["LatestDestinationTime"]) < self.time_horizon:
                    continue

                request = Request(
                    request_id=int(row["RequestID"]),
                    origin_port=row["OriginPort"],
                    destination_port=row["DestinationPort"],
                    earliest_pickup_time=int(row["EarliestPickupTime"]),
                    latest_destination_time=int(row["LatestDestinationTime"]),
                )
                request.number_of_laden_path = int(row["NumberOfLadenPath"])
                request.number_of_empty_path = int(row["NumberOfEmptyPath"])
                if request.number_of_laden_path > 0:
                    request.laden_paths = [
                        self.input_data.container_path_set.get(int(path_id)) for path_id in row["LadenPaths"].split(",")
                        if int(path_id) in self.input_data.container_path_set
                    ]
                    for laden_path in request.laden_paths:
                        if laden_path is None:
                            logger.debug("发现None laden_path")
                        elif not hasattr(laden_path, "path_id"):
                            logger.debug("对象缺少path_id属性：", laden_path)

                    request.laden_path_set = {
                        laden_path.path_id: laden_path
                        for laden_path in request.laden_paths
                        if laden_path is not None and hasattr(laden_path, "path_id")
                    }
                    request.laden_path_indexes = [
                        laden_path.path_id - 1
                        for laden_path in request.laden_paths
                        if laden_path is not None and hasattr(laden_path, "path_id")
                    ]

                if request.number_of_empty_path > 0:
                    request.empty_paths = [
                        self.input_data.container_path_set.get(int(path_id)) for path_id in row["EmptyPaths"].split(",")
                        if int(path_id) in self.input_data.container_path_set
                    ]
                    request.empty_path_set = {empty_path.path_id: empty_path for empty_path in request.empty_paths if empty_path}
                    request.empty_path_indexes = [empty_path.path_id - 1 for empty_path in request.empty_paths if empty_path]
                else:
                    request.empty_paths = []

                if request.earliest_pickup_time <= self.time_horizon and request.latest_destination_time <= self.time_horizon:
                    request_list.append(request)
                    requests[request.request_id] = request
        except Exception as e:
            logger.error(f"error to load requests: {e}")
        logger.info(f"success to load requests")
        self.input_data.requests = request_list
        self.input_data.request_set = requests

    def _read_vessels(self):
        """
        读取船舶数据
        文件格式：VesselID Capacity OperatingCost RouteID MaxNum
        """
        df = self._read_to_dataframe(Config.VESSELS_FILENAME)
        self.input_data.data['vessels'] = df

        vessel_list = []
        vessels = {}
        try:
            for _, row in df.iterrows():
                vessel = VesselType(
                    id=int(row["VesselID"]),
                    capacity=int(row["Capacity"]),
                    cost=float(row["OperatingCost"]) * 1e6,
                    route_id=int(row["RouteID"]),
                    max_num=int(row["MaxNum"])
                )
                vessel_list.append(vessel)
                vessels[vessel.id] = vessel
        except Exception as e:
            logger.error(f"error to load vessels: {e}")
        logger.info(f"success to load vessels")
        self.input_data.vessels = vessel_list
        self.input_data.vessel_type_set = vessels

    def _read_history_solution(self):
        """
        读取历史解决方案数据

        文件格式：
        RequestID VesselPathID Volume
        1 1 20
        1 2 10
        ...
        """
        if not Config.USE_HISTORY_SOLUTION:
            return

        df = self._read_to_dataframe(Config.HISTORY_SOLUTION_FILENAME)
        history_solution = {}

        for _, row in df.iterrows():
            request_id = int(row[0])
            vessel_path_id = int(row[1])
            volume = int(row[2])

            if request_id not in history_solution:
                history_solution[request_id] = {}
            history_solution[request_id][vessel_path_id] = volume

        self.input_data.history_solution_set = history_solution

    def _read_sample_scenes(self):
        """
        读取样本场景数据

        文件格式：
        SceneID RequestID Demand
        1 1 25
        1 2 35
        ...
        """
        if not Config.WHETHER_LOAD_SAMPLE_TESTS:
            return

        df = self._read_to_dataframe(Config.SAMPLE_SCENES_FILENAME)
        sample_scenes = {}

        for _, row in df.iterrows():
            scene_id = int(row[0])
            request_id = int(row[1])
            demand = int(row[2])

            if scene_id not in sample_scenes:
                sample_scenes[scene_id] = {}
            sample_scenes[scene_id][request_id] = demand

        self.input_data.sample_scenes = sample_scenes
def write_df_to_db(df: pd.DataFrame, table_name: str, db_path: str, if_exists: Literal['fail', 'replace', 'append'] = 'replace', index: bool = False):
    """
    将DataFrame写入数据库，自动将列名重命名为README.md标准字段名
    Args:
        df: 待写入的DataFrame
        table_name: 数据库表名（需与DB_COLUMN_MAPS的key一致）
        db_path: 数据库文件路径
        if_exists: 写入模式
        index: 是否写入索引
    """
    # 获取标准字段名映射
    std_map = DB_COLUMN_MAPS.get(table_name, None)
    if std_map:
        # 反向映射：标准字段名 -> 数据库字段名
        db_to_std = {v: v for v in std_map.values()}  # 保证只保留标准字段名
        df = df.rename(columns=db_to_std)
        # 只保留标准字段名列
        df = df[[v for v in std_map.values() if v in df.columns]]
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists=if_exists, index=index)
    conn.close()


if __name__ == '__main__':
    # ================== 用法示例 ==================
    # 读取数据
    data = DataManager()
    reader = DataReader(input_data=data, time_horizon=60)
    reader.read()
