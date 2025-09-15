"""
实体模型包，包含船舶调度与多类型集装箱联合调度问题所需的所有网络相关类。

该包定义了问题中的各种网络组件，包括：
- 港口（Port）
- 船舶路径（VesselPath）及其子类（LadenPath, EmptyPath）
- 请求（Request）
- 船舶航线（ShipRoute）
- 集装箱路径（ContainerPath）
- 起终点范围（ODRange）
"""
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.entity.port import Port
from src.entity.vessel_path import VesselPath
from src.entity.laden_path import LadenPath
from src.entity.empty_path import EmptyPath
from src.entity.request import Request
from src.entity.ship_route import ShipRoute
from src.entity.container_path import ContainerPath
from src.entity.vessel_type import VesselType
from src.entity.od_range import ODRange

__all__ = [
    'Port',
    'VesselPath',
    'LadenPath',
    'EmptyPath',
    'Request',
    'ShipRoute',
    'ContainerPath',
    'VesselType',
    'ODRange'
] 