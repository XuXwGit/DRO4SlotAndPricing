import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
from typing import Any, Dict
from venv import logger
import numpy as np
import os
import sys
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dm.determine_model import DeterministicModel
from src.models.dro.ldr.SOCP4LDR_GRB import SOCP4LDR_GRB
from src.models.dro.ldr.SOCP4LDR_Mosek import SOCP4LDR_Mosek
from src.models.dro.ldr.LDR_model_checker import run_all_validations
from src.utils.data_manager import DataManager
from src.utils.read_data import DataReader
from src.utils.model_params import construct_model_params, generate_feasible_test_case

# ================= 数据读取 =================
# P=20 A = 45 T = 42
# {306: [9, 10, 11, 276, 175, 176], 307: [9, 10, 11, 277, 181, 182], 308: [9, 10, 11, 12, 13, 278, 182], 309: [17, 18, 19, 280, 181, 182], 536: [25, 26], 537: [32, 33], 651: [18, 19, 20, 21, 22], 652: [18, 19, 280, 181, 365, 22], 653: [18, 19, 280, 181, 366, 29], 654: [18, 19, 281, 186, 369, 29], 655: [25, 26, 27, 28, 29], 656: [25, 26, 284, 186, 369, 29], 2378: [15, 16], 2379: [23, 24], 2463: [7, 8, 25, 26], 2464: [15, 16, 32, 33], 2844: [174, 175, 361, 14, 15, 16, 32], 2845: [174, 175, 362, 22, 23, 24, 39], 2846: [174, 359, 20, 21, 22, 23, 24, 39], 2847: [180, 181, 365, 22, 23, 24, 39]}

# ================= 使用示例 =================
if __name__ == "__main__":
    # 从文件中读取算例数据
    logging.debug(f"读取案例数据: ")
    data = DataManager()
    reader = DataReader(input_data=data, time_horizon=60)
    reader.read(case="2")
    data.generate_demand_and_price()
    # TODO 按照 construct_model_params 方法构造的参数，无法生成一个最优解，存在数值问题
    model_params = construct_model_params(data_manager=data)
    # 仅保留读取的 路径和边/Deadline 的集合
    model_params = generate_feasible_test_case(
                # num_paths=30,
                # paths = {306: [9, 10, 11, 276, 175, 176], 307: [9, 10, 11, 277, 181, 182], 308: [9, 10, 11, 12, 13, 278, 182], 309: [17, 18, 19, 280, 181, 182], 536: [25, 26], 537: [32, 33], 651: [18, 19, 20, 21, 22], 652: [18, 19, 280, 181, 365, 22], 653: [18, 19, 280, 181, 366, 29], 654: [18, 19, 281, 186, 369, 29], 655: [25, 26, 27, 28, 29], 656: [25, 26, 284, 186, 369, 29], 2378: [15, 16], 2379: [23, 24], 2463: [7, 8, 25, 26], 2464: [15, 16, 32, 33], 2844: [174, 175, 361, 14, 15, 16, 32], 2845: [174, 175, 362, 22, 23, 24, 39], 2846: [174, 359, 20, 21, 22, 23, 24, 39], 2847: [180, 181, 365, 22, 23, 24, 39]},
                paths=model_params['paths'],
                deadline=model_params['t_d_phi'],
                num_periods=42,
                num_prices=10,
                uncertainty_dim=5,
                uncertainty_std_ratio=1.0,                 # Feasible range: [0.01, 0.5, 1.0]
                base_mean_demand=100,                    # Feasible range: [30, 100]
                price_range=(500, 1000),                   # Feasible range: [10, 100] 
                demand_sensitivity=1,
                base_capacity=5000,                          # Feasible range: [500, 5000]
                seed=42  # 固定种子以复现结果
            )


    print("\n--- 求解 LP Relaxation 版本 ---")
    det_model_lp = DeterministicModel(model_params, use_lp_relaxation=True)
    det_model_lp.build_model()
    det_model_lp.solve()
    det_model_lp.write()
    # print(det_model_lp.solutions)
    if det_model_lp.get_status() != 'OPTIMAL':
        print(det_model_lp.get_status())

    print("\n--- 求解 DRO 版本 ---")
    # socp = SOCP4LDR_GRB(model_params=model_params)
    socp = SOCP4LDR_Mosek(model_params=model_params)
    socp.build_model()
    socp.solve()
    print(socp.get_status())
    if socp.get_status() != 'OPTIMAL':
        print(socp.get_status())

    socp.write()
    socp.get_solution()

    # check feasibility
    run_all_validations(socp.get_solution(), model_params, tolerance=1e-2)
