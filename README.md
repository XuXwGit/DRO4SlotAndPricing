# DRO4SlotAndPricing

DRO4SlotAndPricing 是一个针对班轮运输集装箱舱位分配与定价问题的研究代码仓库，聚焦于基于分布鲁棒优化（Distributionally Robust Optimization, DRO）的建模与求解方法。项目实现了数据加载、网络构建、模型生成及求解等关键流程，便于进一步的算法实验与性能评估。

## 项目结构
- `src/config/`：集中管理求解器、实验设置与数据路径等运行时配置。
- `src/entity/`：定义港口、航线、路径、需求等领域对象的数据结构。
- `src/network/`：封装时空网络节点、航行弧和转运弧等结构化网络组件。
- `src/models/`：包含基础模型构建器以及面向 DRO、SOCP 等不同求解策略的实现。
- `src/utils/`：提供数据读取、预处理与辅助工具（如 `DataReader` 与 `DataManager`）。
- `data/`：存放算例输入数据以及字段规范说明。

## 环境准备
1. 建议使用 Python 3.10+ 创建虚拟环境：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   ```
2. 安装依赖（示例）：
   ```bash
   pip install pandas numpy gurobipy
   ```
   若需使用 Mosek 等其他求解器，请参照相应官方文档额外安装。

## 快速开始
以下示例展示了如何读取算例数据并构建基础模型实例：
```python
from src.utils.data_manager import DataManager
from src.utils.read_data import DataReader
from src.models.model_builder import ModelBuilder

manager = DataManager()
reader = DataReader(input_data=manager, time_horizon=60)
manager = reader.read(case="1")

model = ModelBuilder(info="baseline", solver="gurobi")
model.build_model()
model.solve()
```
数据文件说明请参考 `data/README.md`，其中包含各类输入文件的字段规范与算例规模统计。

## 许可
本仓库未附带显式开源协议，使用前请与作者确认相关权益。
