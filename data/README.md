# 数据文件表头与数据库列名统一规范

## 1. 数据分类

- **基础网络数据**
  - Ports（港口）
  - ShippingRoutes / Routes（航线）
  - Vessels（船舶）

- **时空网络数据**
  - Nodes（时空节点）
  - TravelingArcs（航行弧）
  - TransshipArcs（转运弧）
  - Paths（路径）
  - VesselPaths（航行往返路径）
  - LadenPaths（重载路径）
  - EmptyPaths（空载路径）

- **运输需求数据**
  - DemandRange（需求区间）
  - Requests（运输需求）
  - DemandsSXX-TXX(Uniform)（多场景需求）

---

## 2. 各类数据文件表头与数据库列名规范

### Ports.txt
| 字段         | 字段说明     | 类型   | 数据库字段    |
|--------------|--------------|--------|--------------|
| PortID       | 港口编号     | int    | port_id      |
| Port         | 港口名称     | str    | port_name    |
| WhetherTrans | 是否为中转港 | int    | is_trans     |
| Region       | 区域         | str    | region       |
| Group        | 分组         | int    | group_id     |

**示例：**

| PortID | Port | WhetherTrans | Region | Group |
|--------|------|--------------|--------|-------|
| 1      | A    | 0            | I      | 1     |

---

### ShippingRoutes.txt
| 字段            | 字段说明     | 类型   | 数据库字段        |
|-----------------|--------------|--------|-------------------|
| ShippingRouteID | 航线编号     | int    | shipping_route_id |
| NumberOfPorts   | 港口数量     | int    | num_ports         |
| Ports           | 港口序列     | str    | ports             |
| NumberOfCall    | 挂靠次数     | int    | num_calls         |
| PortsOfCall     | 挂靠港口序列 | str    | ports_of_call     |
| Time            | 挂靠时刻序列 | str    | call_times        |

**示例：**

| ShippingRouteID | NumberOfPorts | Ports           | NumberOfCall | PortsOfCall        | Time                   |
|-----------------|--------------|-----------------|--------------|--------------------|------------------------|
| 1               | 7            | A,B,C,D,E,F,G   | 8            | A,B,C,D,E,F,G,A    | 2,3,4,10,12,14,17,23   |

---

### Vessels.txt
| 字段        | 字段说明 | 类型   | 数据库字段   |
|-------------|----------|--------|-------------|
| VesselID    | 船舶编号 | int    | vessel_id   |
| Capacity    | 船舶容量 | int    | capacity    |
| OperatingCost | 运营成本 | float  | operating_cost |
| RouteID     | 航线编号 | int    | route_id    |
| MaxNum      | 最大数量 | int    | max_num     |

**示例：**

| VesselID | Capacity | OperatingCost | RouteID | MaxNum |
|----------|----------|---------------|---------|--------|
| 1        | 3168     | 10.12         | 1       | 1      |

---

### Nodes.txt
| 字段      | 字段说明 | 类型   | 数据库字段   |
|-----------|----------|--------|-------------|
| ID        | 节点编号 | int    | node_id     |
| Route     | 航线编号 | int    | route_id    |
| Call      | 挂靠序号 | int    | call        |
| Port      | 港口名称 | str    | port_name   |
| RoundTrip | 航次     | int    | round_trip  |
| Time      | 时刻     | int    | time        |

**示例：**

| ID | Route | Call | Port | RoundTrip | Time |
|----|-------|------|------|-----------|------|
| 1  | 1     | 1    | A    | 1         | 2    |

---

### TravelingArcs.txt
| 字段              | 字段说明     | 类型   | 数据库字段         |
|-------------------|--------------|--------|--------------------|
| TravelingArcID    | 航行弧编号   | int    | traveling_arc_id   |
| Route             | 航线编号     | int    | route_id           |
| OriginNodeID      | 起点节点编号 | int    | origin_node_id     |
| OriginCall        | 起点挂靠序号 | int    | origin_call        |
| OriginPort        | 起点港口     | str    | origin_port        |
| RoundTrip         | 航次         | int    | round_trip         |
| OriginTime        | 起点时刻     | int    | origin_time        |
| TravelingTime     | 航行时间     | int    | traveling_time     |
| DestinationNodeID | 终点节点编号 | int    | destination_node_id|
| DestinationCall   | 终点挂靠序号 | int    | destination_call   |
| DestinationPort   | 终点港口     | str    | destination_port   |
| DestinationTime   | 终点时刻     | int    | destination_time   |

**示例：**

| TravelingArcID | Route | OriginNodeID | OriginCall | OriginPort | RoundTrip | OriginTime | TravelingTime | DestinationNodeID | DestinationCall | DestinationPort | DestinationTime |
|----------------|-------|-------------|------------|------------|-----------|------------|---------------|-------------------|-----------------|-----------------|-----------------|
| 1              | 1     | 1           | 1          | A          | 1         | 2          | 1             | 2                | 2               | B              | 3               |

---

### TransshipArcs.txt
| 字段            | 字段说明     | 类型   | 数据库字段         |
|-----------------|--------------|--------|--------------------|
| TransshipArcID  | 转运弧编号   | int    | transship_arc_id   |
| Port            | 港口         | str    | port_name          |
| OriginNodeID    | 起点节点编号 | int    | origin_node_id     |
| OriginTime      | 起点时刻     | int    | origin_time        |
| TransshipTime   | 转运时间     | int    | transship_time     |
| DestinationNodeID | 终点节点编号 | int  | destination_node_id|
| DestinationTime | 终点时刻     | int    | destination_time   |
| FromRoute       | 来源航线     | int    | from_route         |
| ToRoute         | 去向航线     | int    | to_route           |

**示例：**

| TransshipArcID | Port | OriginNodeID | OriginTime | TransshipTime | DestinationNodeID | DestinationTime | FromRoute | ToRoute |
|----------------|------|-------------|------------|---------------|-------------------|-----------------|-----------|---------|
| 272            | D    | 4           | 10         | 2             | 172               | 12              | 1         | 2       |

---


### VesselPaths
| 字段名（英文） | 字段名（中文） | 类型 | 说明 |
|--------------------|---------------|---------|------------------------------|
| VesselPathID | 路径ID | int | 航行轮回路径唯一编号 |
| ShippingRouteID | 航线ID | int | 该路径所属的航线编号 |
| NumberOfArcs | 弧段数 | int | 路径包含的弧段数量 |
| ArcIDs | 弧段ID序列 | string | 该路径包含的弧段ID（逗号分隔）|
| OriginTime | 起点时间 | int | 路径起点节点的时间 |
| DestinationTime | 终点时间 | int | 路径终点节点的时间 |

**示例：**
VesselPathID	ShippingRouteID	NumOfArcs	ArcIDs	OriginTime	DestinationTime
1	1	14	1,2,3,4,5,6,7,8,9,10,11,12,13,14	3	87

---

### Paths.txt
| 字段            | 字段说明     | 类型   | 数据库字段         |
|-----------------|--------------|--------|--------------------|
| ContainerPathID | 路径编号     | int    | container_path_id  |
| OriginPort      | 起点港口     | str    | origin_port        |
| OriginTime      | 起点时刻     | int    | origin_time        |
| DestinationPort | 终点港口     | str    | destination_port   |
| DestinationTime | 终点时刻     | int    | destination_time   |
| PathTime        | 路径用时     | int    | path_time          |
| TransshipPort   | 中转港口     | int    | transship_port     |
| TransshipTime   | 中转用时     | int    | transship_time     |
| PortPathLength  | 港口序列长度 | int    | port_path_length   |
| PortPath        | 港口序列     | str    | port_path          |
| ArcsLength      | 弧段数       | int    | arcs_length        |
| ArcsID          | 弧段编号     | str    | arcs_id            |

**示例：**

| ContainerPathID | OriginPort | OriginTime | DestinationPort | DestinationTime | PathTime | TransshipPort | TransshipTime | PortPathLength | PortPath | ArcsLength | ArcsID |
|-----------------|------------|------------|-----------------|-----------------|----------|---------------|---------------|---------------|----------|------------|--------|
| 1               | A          | 2          | B               | 3               | 1        | 0             | 0             | 2             | A,B      | 1          | 1      |

---

### LadenPaths.txt
| 字段            | 字段说明         | 类型   | 数据库字段         |
|-----------------|------------------|--------|--------------------|
| RequestID       | 需求编号         | int    | request_id         |
| OriginPort      | 起点港口         | str    | origin_port        |
| OriginTime      | 起点时刻         | int    | origin_time        |
| DestinationPort | 终点港口         | str    | destination_port   |
| RoundTrip       | 航次             | int    | round_trip         |
| EarliestPickupTime      | 最早发运时刻     | int    | wi_earliest        |
| ArrivalTimeToDestination | 到达终点时刻 | int | arrival_time_to_destination |
| PathTime        | 路径用时         | int    | path_time          |
| TransshipPort   | 中转港口         | int    | transship_port     |
| TransshipTime   | 中转用时         | int    | transship_time     |
| PathID          | 路径编号         | int    | path_id            |
| PortPath        | 港口序列         | str    | port_path          |
| ArcIDs          | 弧段编号         | str    | arc_ids            |

**示例：**
| RequestID | OriginPort | OriginTime | DestinationPort | RoundTrip | EarliestPickupTime | ArrivalTimeToDestination | PathTime | TransshipPort | TransshipTime | PathID | PortPath | ArcIDs |
|-----------|------------|------------|-----------------|-----------|------------|-------------------------|----------|---------------|---------------|--------|----------|--------|
| 1         | A          | 2          | B               | 1         | 1          | 3                       | 1        | 0             | 0             | 1      | A,B      | 1      |

---

### EmptyPaths.txt
| 字段            | 字段说明         | 类型   | 数据库字段         |
|-----------------|------------------|--------|--------------------|
| RequestID       | 需求编号         | int    | request_id         |
| OriginPort      | 起点港口         | str    | origin_port        |
| OriginTime      | 起点时刻         | int    | origin_time        |
| NumOfEmptyPath  | 空载路径数       | int    | num_of_empty_path  |
| PathIDs         | 路径编号集合     | str    | path_ids           |

**示例：**
| RequestID | OriginPort | OriginTime | NumOfEmptyPath | PathIDs |
|-----------|------------|------------|----------------|---------|
| 1         | A          | 1          | 0              | 0       |


---

### Requests.txt
| 字段              | 字段说明     | 类型   | 数据库字段         |
|-------------------|--------------|--------|--------------------|
| RequestID         | 需求编号     | int    | request_id         |
| OriginPort        | 起点港口     | str    | origin_port        |
| DestinationPort   | 终点港口     | str    | destination_port   |
| EarliestPickupTime        | 最早发运时刻 | int    | wi_earliest        |
| LatestDestinationTime | 最晚到达时刻 | int | latest_destination_time |
| LadenPaths        | 可选重载路径 | str    | laden_paths        |
| NumberOfLadenPath | 重载路径数   | int    | number_of_laden_path|
| EmptyPaths        | 可选空载路径 | str    | empty_paths        |
| NumberOfEmptyPath | 空载路径数   | int    | number_of_empty_path|

**示例：**

| RequestID | OriginPort | DestinationPort | EarliestPickupTime | LatestDestinationTime | LadenPaths | NumberOfLadenPath | EmptyPaths | NumberOfEmptyPath |
|-----------|------------|----------------|------------|----------------------|------------|-------------------|------------|-------------------|
| 1         | A          | B              | 1          | 15                   | 1,2        | 2                 | 0          | 0                 |

---

### DemandRange.txt
| 字段              | 字段说明     | 类型   | 数据库字段         |
|-------------------|--------------|--------|--------------------|
| OriginRegion      | 起点区域     | int    | origin_region      |
| DestinationRegion | 终点区域     | int    | destination_region |
| DemandLowerBound  | 需求下界     | int    | demand_lower_bound |
| DemandUpperBound  | 需求上界     | int    | demand_upper_bound |
| FreightLowerBound | 运价下界     | int    | freight_lower_bound|
| FreightUpperBound | 运价上界     | int    | freight_upper_bound|

**示例：**

| OriginRegion | DestinationRegion | DemandLowerBound | DemandUpperBound | FreightLowerBound | FreightUpperBound |
|--------------|------------------|------------------|------------------|-------------------|-------------------|
| 1            | 2                | 20               | 40               | 3500              | 5500              |

---

### DemandsSXX-TXX(Uniform).txt
| 字段名    | 说明     | 数据类型 | 数据库列名 |
|-----------|----------|----------|------------|
| SceneID   | 场景编号 | int      | scene_id   |
| RequestID | 需求编号 | int      | request_id |
| Demand    | 需求量   | int      | demand     |
**示例：**
| SceneID	| RequestID	| Demand |
|---------|-----------|--------|
| 1	| 1001	| 15 |

---

## 3. 各算例数据规模统计

| 算例 | 港口数 | 船舶数 | 节点数 | 航行弧数 | 转运弧数 | 路径数 | 需求数 | 备注 |
|------|--------|--------|--------|----------|----------|--------|--------|------|
| 1    | 10     | 14     | 280    | 273      | 169      | 3795   | 2342   |      |
| 2    | 29     | 40     | 926    | 892      | 1344     | 22859  | 6111   |      |
| 3    | 22     | 36     | 593    | 558      | 1171     | -      | -      |      |
| 4    | 23     | 37     | 595    | 558      | 1171     | -      | -      |      |
| 5    | 22     | 17     | 595    | 580      | 954      | -      | -      |      |

> 注：部分算例路径、需求等文件未完全统计，若需详细补充可进一步读取。

---

**所有后续开发、数据导入、数据库建表、前后端交互等，均以本规范为唯一标准。**
