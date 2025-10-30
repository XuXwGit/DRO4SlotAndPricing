# DRO4SlotAndPricing

DRO4SlotAndPricing 是一项利用分布鲁棒优化（Distributionally Robust Optimization, DRO）方法求解班轮运输的集装箱舱位分配与定价问题的研究项目的相关代码，提供从数据加载到模型构建、求解的完整实验代码。

## 快速开始
1. 建议使用 Python 3.10+ 创建虚拟环境并安装依赖：
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 使用 .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. 准备输入数据：`data/` 目录已按算例编号归档，字段规范详见 `data/README.md`。
3. 构建并求解基础模型示例：
   ```python
   from src.utils.data_manager import DataManager
   from src.utils.read_data import DataReader
   from src.models.model_builder import ModelBuilder

   manager = DataManager()
   reader = DataReader(manager, time_horizon=60)
   data = reader.read(case="1")

   model = ModelBuilder(info="baseline", solver="gurobi")
   model.build_model()
   model.solve()
   ```

## 可视化运行面板
项目提供了一个基于 Flask 的前端页面，可视化配置模型参数并触发求解：

```bash
python -m src.frontend.app
```

默认在 <http://127.0.0.1:5000> 提供服务。页面支持：
- 随机生成或基于案例数据构造模型参数；
- 选择求解确定性模型或 DRO 模型（若环境已安装对应求解器）；
- 以表格和变量快照展示求解状态、目标值及关键统计信息。

若求解器未安装或许可不可用，页面会在结果区域提示错误详情。

## 目录速览
- `src/config/`：全局路径、求解器参数等运行配置。
- `src/entity/`：港口、航线、路径、需求等领域实体定义。
- `src/network/`：时空网络节点、航行弧与转运弧数据结构。
- `src/models/`：模型基类及 DRO/SOCP 等求解策略实现。
- `src/utils/`：数据读取、管理及辅助工具函数。
- `src/frontend/`：可视化前端应用与模型调度逻辑。
- `data/`：算例原始文件及字段规范文档。

## 数据文档
- 欲了解每个输入文件的字段、示例及规模统计，请阅读 `data/README.md`。

## 版本记录
- v0.3：精简 README，补充快速上手步骤并强调数据文档位置。
- v0.4：新增 Flask 前端运行面板并整合依赖安装指引。
