"""
网络模型包，包含时空网络相关类。

该包定义了问题中的各种网络组件，包括：
- 节点（Node）
- 弧（Arc）及其子类（TravelingArc, TransshipArc）
"""

from network.node import Node
from network.arc import Arc
from network.traveling_arc import TravelingArc
from network.transship_arc import TransshipArc

__all__ = [
    'Node',
    'Arc',
    'TravelingArc',
    'TransshipArc',
] 