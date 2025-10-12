import logging
from typing import Any, Dict
from venv import logger
import numpy as np
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.dro.ldr.SOCP4LDR_GRB import SOCP4LDR
from src.models.dro.ldr.SOCP4LDR_Mosek import SOCP4LDR_Mosek
from src.models.dro.ldr.LDR_model_checker import run_all_validations
from src.utils.data_manager import DataManager
from src.utils.read_data import DataReader
from src.utils.model_params import construct_model_params, generate_feasible_test_case

# ================= 使用示例 =================
if __name__ == "__main__":
    try:
        # 从文件中读取算例数据
        data = DataManager()
        reader = DataReader(input_data=data, time_horizon=30)
        reader.read()
        data.generate_demand_and_price()
        model_params = construct_model_params(data_manager=data)
    except Exception as e:
        logging.debug(f"读取案例数据 or 生成模型参数 失败: {e}")
        # 兜底：生成一个测试用例
        model_params = generate_feasible_test_case(
            num_paths=10,
            num_periods=10,
            num_prices=10,
            uncertainty_dim=1,
            seed=42  # 固定种子以复现结果
        )

    model_params = generate_feasible_test_case(
            num_paths=1,
            num_periods=2,
            num_prices=1,
            uncertainty_dim=1,
            seed=42  # 固定种子以复现结果
        )


    # socp = SOCP4LDR(model_params=model_params)
    socp = SOCP4LDR_Mosek(model_params=model_params)

    socp.build_model()

    # 求解
    socp.solve()
    print(socp.get_status())

    # check feasibility
    # run_all_validations(socp.get_solution(), model_params, socp.get_pi_solution())
