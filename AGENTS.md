# Project Guide

## 项目概述
- **名称**：DRO4SlotAndPricing（班轮运输配载与定价分布鲁棒优化）
- **目标**：基于分布鲁棒优化 (Distributionally Robust Optimization, DRO) 构建班轮运输的舱位分配与定价模型，支持确定性与随机情景下的容量规划与收益管理。
- **核心特性**：
  - 通过 `src/models` 下的构建器组合确定性模型、SOCP 松弛以及 LDR (Linear Decision Rule) 的 DRO 模型。
  - 使用 `src/utils` 中的数据管理与参数生成工具，连接 `data/` 中的案例输入，形成可求解的数学规划模型。
  - 网络结构由 `src/network`（节点、弧与转运结构）和 `src/entity`（船舶、航线、请求、路径等领域对象）模块抽象。

## 运行与依赖
- **语言**：Python 3.10+
- **数学规划求解器**：默认集成 [Gurobi](https://www.gurobi.com/)；部分 DRO 实现支持 [MOSEK](https://www.mosek.com/)。确保本地已安装并授权相关求解器。
- **主要第三方库**：
  - `gurobipy`（数学规划建模与求解）
  - `numpy`（数值运算与矩阵操作）
  - `logging`（统一日志输出）
  - 其他依赖请参考项目的 `requirements`（若未提供，请使用 `pip list` 确认环境）。
- **数据**：`data/` 目录按照案例编号区分，测试脚本默认读取目录 `data/2/` 等文件。

## 目录速览
- `src/config/`：系统与数据配置（求解参数、随机参数、日志开关等）。
- `src/entity/`：业务实体定义（港口、路径、请求、船型、场景等）。
- `src/network/`：网络图结构（节点、旅行弧、转运弧）。
- `src/models/`：模型基类与不同求解策略（`model_builder.py`、`SOCP_model_builder.py`、`dm`、`dro` 等子包）。
- `src/utils/`：数据读取、参数构造、辅助工具。
- `src/test/`：示例测试脚本，展示从数据到模型的端到端流程。

## 编程规范
- **命名风格**：Python 代码遵循 PEP 8；类名使用 `CamelCase`，函数与变量使用 `snake_case`。常量与配置项使用全大写蛇形命名。
- **类型标注**：优先为公共接口、复杂数据结构添加类型注解（见 `src/network/arc.py` 等示例）。
- **日志与调试**：统一使用 `logging` 模块；调试开关由 `Config.debug_mode` 等配置控制，避免在业务逻辑中直接使用 `print`。
- **异常处理**：核心流程（如模型构建、求解）使用 `try/except` 捕获并记录日志，同时保持异常向上抛出，以便调用方感知失败。
- **文档与注释**：模块与类顶部采用三引号文档字符串，简要说明作者、用途、日期；公共方法补充 docstring，便于理解参数与返回值。
- **依赖管理**：新增依赖需在 `requirements.txt`（若创建）或相关文档中说明；涉及商业求解器的接口须提供替代路径或清晰的环境准备指南。

## 开发建议
- 修改求解参数时请同步更新 `src/config/config.py` 中的默认值，确保测试脚本能读取一致配置。
- 扩展 DRO 模型或新增求解器适配层时，优先继承 `ModelBuilder` 或 `SOCPModelBuilder`，实现统一的 `build_model → solve → extract_solution` 流程。
- 新增数据字段需同时更新 `DataManager`、`construct_model_params`、`generate_feasible_test_case` 等数据管线模块。
- 提交前至少运行一次 `python src/test/test_LDR_SOCP.py` 或自定义测试用例，验证模型构建与求解是否成功。

## 提交说明
- 保持 Git 历史整洁，针对功能或修复创建独立分支。
- 代码变更需附带必要的单元/集成测试及文档更新。
- 提交信息应清晰描述修改内容与影响范围。

## 当前 TODO List（版本 2）
当前正在开发数值试验模块，主要包括：
1. 灵敏度分析：固定真实数据集，在不同 sigma_sq、d_z 放大数、a 等参数下比较确定性模型 vs DRO-LDR 目标值、Y/X 差异。
2. 规模鲁棒性：构造不同路径/时段规模的随机算例，记录求解时间、目标值等。
3. 不同模糊集：针对构造的案例，改变模糊集合的定义（如 MM、PCM）。
4. 决策规则检验：在 LDR 解基础上模拟多轮需求实现（蒙特卡洛），评估不同策略。

