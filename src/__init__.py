# src/__init__.py

"""
Main package for the slot allocation and dynamic pricing optimization project.
"""

# 导入常用配置
from .config import config
from .config import data_config

# 导入核心实体类（按需暴露）
from .entity import (
    Port,
    VesselType,
    ShipRoute,
    Scenario,
    Request,
    ContainerPath,
    LadenPath,
    EmptyPath,
    VesselPath,
    ODRange,
)

# 导入网络组件
from .network import (
    Node,
    Arc,
    TravelingArc,
    TransshipArc,
)

# 导入模型构建器
from .models.model_builder import ModelBuilder

# 导入工具函数
from .utils import (
    read_data,
    data_manager,
    model_params,
)

# 可选：导入测试用例（一般不推荐在 __init__ 中导入 test）
# 保持干净，测试模块通常由 pytest 等单独运行

# 定义 __all__ 以明确公开接口（可选但推荐）
__all__ = [
    # Config
    'config',
    'data_config',

    # Entities
    'Port',
    'VesselType',
    'ShipRoute',
    'Scenario',
    'Request',
    'ContainerPath',
    'LadenPath',
    'EmptyPath',
    'VesselPath',
    'ODRange',

    # Network
    'Node',
    'Arc',
    'TravelingArc',
    'TransshipArc',

    # Models
    'ModelBuilder',

    # Utils
    'read_data',
    'data_manager',
    'model_params',
]
