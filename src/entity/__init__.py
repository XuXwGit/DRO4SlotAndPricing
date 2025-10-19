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

from .port import Port
from .vessel_type import VesselType
from .ship_route import ShipRoute
from .scenario import Scenario
from .request import Request
from .container_path import ContainerPath
from .laden_path import LadenPath
from .empty_path import EmptyPath
from .vessel_path import VesselPath
from .od_range import ODRange

__all__ = [
    'Port', 'VesselType', 'ShipRoute', 'Scenario', 'Request',
    'ContainerPath', 'LadenPath', 'EmptyPath', 'VesselPath', 'ODRange'
]
