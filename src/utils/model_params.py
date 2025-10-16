from random import random
from typing import Any, Dict

import numpy as np
from utils.data_manager import DataManager


def construct_model_params(data_manager: 'DataManager') -> Dict[str, Any]:
    """
    从 DataManager 实例中提取并构造 model_params 字典。

    该字典包含了 SOCP4LDR 模型和验证函数所需的全部参数。

    Args:
        data_manager: 已经加载了数据的 DataManager 实例。

    Returns:
        Dict[str, Any]: 包含模型参数的字典。
    """
    try:
        # 1. 提取路径 (phi_list)
        # 在模型中，路径 phi 对应于 container_paths (集装箱路径)
        # phi_list = [cp.container_path_id for cp in data_manager.container_paths]
        # paths = {cp.container_path_id: cp.arcs_id for cp in data_manager.container_paths}
        phi_list = []
        paths = {}
        for request in data_manager.requests:
            # 假设 Request 类有一个 price_set 属性，它是一个列表
            for path in request.laden_path_set.values():
                phi_list.append(path.id)
                paths[path.id] = path.arcs_id
        A_prime = {arc.id: arc.capacity for arc in data_manager.arcs}

        # 2. 提取时间段 (t_list)
        t_list = list(range(1, data_manager.time_horizon + 1))

        # 3. 提取价格点 (p_list)
        p_list = {}
        for request in data_manager.requests:
            # 假设 Request 类有一个 price_set 属性，它是一个列表
            for path in request.laden_path_set.values():
                p_list[path.id] = getattr(request, 'price_set', [request.long_haul_price * (1 + 0.1 * i) - path.path_cost for i in range(10)])

        # 4. 提取基准价格 p_hat
        p_hat = {}
        for request in data_manager.requests:
            for path in request.laden_path_set.values():
                p_hat[path.id] = request.long_haul_price - path.path_cost

        # 7. 提取基础需求 d_0_phi_t
        d_0_phi_t = {}
        for request in data_manager.requests:
            d0 = getattr(request, 'mean_demand', 10.0) + getattr(request, 'variance_demand', 10.0)
            for cp in request.laden_path_set.values():
                for t in t_list:
                    d_0_phi_t[(cp.container_path_id, t)] = d0

        # 6. 提取需求有效期 t_d_phi
        t_d_phi = {}
        for cp in data_manager.container_paths:
            t_d_phi[cp.container_path_id] = getattr(cp, 'origin_time', data_manager.time_horizon)

        # 9. 提取价格敏感度 a
        a = getattr(data_manager, 'price_sensitivity', 1.0)

        # 5. 提取 per-period  revenue:  c_phi_tp
        # c_φtp = p Kφt = p dφt − a(p − pˆφ) = p dφt − a p^2 + a p pˆφ
        c_phi_tp = {}
        for request in data_manager.requests:
            for cp in request.laden_path_set.values():
                for t in t_list:
                    for p in p_list:
                        if t < t_d_phi[cp.container_path_id]:
                            c_phi_tp[(cp.container_path_id, t, p)] = p * d_0_phi_t[(cp.container_path_id, t)] - a * p * p + a * p * p_hat[cp.container_path_id]
                        else:
                            c_phi_tp[(cp.container_path_id, t, p)] = 0

        # 8. 提取需求对 z 的敏感度 d_z_phi_t_i
        # 不确定性维度 I1 设为港口数量
        I1 = len(data_manager.port_set) if data_manager.port_set else 2
        d_z_phi_t_i = {}
        for cp in data_manager.container_paths:
            for t in t_list:
                for i in range(I1):
                    # 为简化，假设一个固定值。在实际应用中，这可能基于路径是否经过第 i 个港口
                    d_z_phi_t_i[(cp.container_path_id, t, i)] = -0.5


        # 10. 提取不确定性参数 (mu, sigma_sq, Sigma)
        # 这些参数通常在 DataManager 的 data 属性或特定字段中
        mu = data_manager.data.get('mu', [1.0] * I1)
        sigma_sq = data_manager.data.get('sigma_sq', [1.0] * I1)
        Sigma = data_manager.data.get('Sigma', [[1.0 if i == j else 0.0 for j in range(I1)] for i in range(I1)])

        # 11. 构建 model_params 字典
        model_params = {
            'I1': I1,
            'phi_list': phi_list,
            't_list': t_list,
            'p_list': p_list,
            'p_hat': p_hat,
            'c_phi_tp': c_phi_tp,
            't_d_phi': t_d_phi,
            'd_0_phi_t': d_0_phi_t,
            'd_z_phi_t_i': d_z_phi_t_i,
            'a': a,
            'mu': mu,
            'sigma_sq': sigma_sq,
            'Sigma': Sigma,
            "paths": paths,
            "A_prime": A_prime,
        }
    except Exception as e:
        print(f"Error constructing model parameters: {e}")
        raise e

    return model_params


def generate_feasible_test_case(
    num_paths=2,
    num_periods=3,
    num_prices=3,
    uncertainty_dim=2,
    base_demand_range=(500.0, 1000.0),  # ← 减小上限
    price_range=(100.0, 210.0),
    base_price_ratio=0.5,
    cost_ratio=0.8,
    demand_sensitivity=0.1,
    uncertainty_std_ratio=0.0,
    seed=42
):
    if seed is not None:
        np.random.seed(seed)

    # --- 1. 基础集合 ---
    phi_list = [f"P{i+1}" for i in range(num_paths)]
    t_list = list(range(0, num_periods + 1))
    p_list = np.linspace(price_range[0], price_range[1], num_prices).tolist()
    p_min = min(p_list)

    # --- 2. 路径参数 ---
    p_hat = {}
    t_d_phi = {}
    max_demand = 0.0
    for phi in phi_list:
        p_hat[phi] = p_min * base_price_ratio
        t_d_phi[phi] = np.random.randint(max(1, num_periods - 1), num_periods)

    # --- 3. 基础需求 (动态跟踪最大需求) ---
    d_0_phi_t = {}
    for phi in phi_list:
        for t in t_list:
            if t <= t_d_phi[phi]:
                d_val = np.random.uniform(base_demand_range[0], base_demand_range[1])
                d_0_phi_t[(phi, t)] = d_val
                max_demand = max(max_demand, d_val)
            else:
                d_0_phi_t[(phi, t)] = 0.0

    # --- 4. 动态设置容量 (确保容量 > 最大需求) ---
    safety_factor = 1.5  # 安全余量
    paths = {}
    A_prime = {}
    for i, phi in enumerate(phi_list):
        edge = (f"N{i}", f"N{i+1}")
        paths[phi] = [edge]
        # 容量 = 最大需求 × 安全因子
        A_prime[edge] = max_demand * safety_factor

    # --- 5. 成本系数 ---
    c_phi_tp = {}
    for phi in phi_list:
        for t in t_list:
            for p in p_list:
                c_phi_tp[(phi, t, p)] = p * cost_ratio

    # --- 6. 不确定性参数 (减小方差) ---
    mu = np.zeros(uncertainty_dim)
    # 方差 = (最大需求 × std_ratio)^2
    sigma_sq = np.array([
        (max_demand * uncertainty_std_ratio) ** 2
        for _ in range(uncertainty_dim)
    ])
    Sigma = np.diag(sigma_sq)

    # --- 7. 需求波动 ---
    d_z_phi_t_i = {}
    for phi in phi_list:
        for t in t_list:
            for i in range(uncertainty_dim):
                # 波动与基础需求成比例
                d_z_phi_t_i[(phi, t, i)] = 0.1 * d_0_phi_t[(phi, t)]

    # --- 8. 返回结果 ---
    test_case = {
        "I1": uncertainty_dim,
        "phi_list": phi_list,
        "t_list": t_list,
        "p_list": p_list,
        "p_hat": p_hat,
        "c_phi_tp": c_phi_tp,
        "t_d_phi": t_d_phi,
        "mu": mu,
        "sigma_sq": sigma_sq,
        "Sigma": Sigma,
        "d_0_phi_t": d_0_phi_t,
        "d_z_phi_t_i": d_z_phi_t_i,
        "a": demand_sensitivity,
        "paths": paths,
        "A_prime": A_prime,
        "cost_cov": np.sum(Sigma)  # 添加 cost_cov
    }

    print(f"  路径数: {num_paths}, 时段数: {num_periods}")
    print(f"  最大需求: {max_demand:.2f}, 容量: {A_prime[list(A_prime.keys())[0]]:.2f}")
    print(f"  方差 sigma_sq: {sigma_sq[0]:.2f}")
    return test_case
