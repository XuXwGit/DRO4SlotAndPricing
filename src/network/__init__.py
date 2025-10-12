"""
网络模型包，包含时空网络相关类。

# multi.network 说明

本包定义了船舶调度与集装箱运输网络的基础结构，包括节点、弧、网络拓扑等。

## 主要类说明

- `Node`：网络节点类，描述港口、时间点等网络节点。
- `Arc`：网络弧类，描述运输、转运等网络连接。
- `TravelingArc`/`TransshipArc`：具体运输弧/转运弧的实现。
---
详细结构和接口请查阅各类源代码及注释。
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
