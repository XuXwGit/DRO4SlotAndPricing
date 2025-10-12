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
    # 1. 提取路径 (phi_list)
    # 在模型中，路径 phi 对应于 container_paths (集装箱路径)
    phi_list = [cp.container_path_id for cp in data_manager.container_paths]
    paths = {cp.container_path_id: cp.arcs_id for cp in data_manager.container_paths}
    A_prime = {arc.id: arc.capacity for arc in data_manager.arcs}

    # 2. 提取时间段 (t_list)
    t_list = list(range(1, data_manager.time_horizon + 1))

    # 3. 提取价格点 (p_list)
    p_list = {}
    # Todo 修改为每个 path 一个 price 列表
    for request in data_manager.requests:
        # 假设 Request 类有一个 price_set 属性，它是一个列表
        for path in request.laden_path_set.values():
            p_list[path.id] = getattr(request, 'price_set', [request.long_haul_price * (1 + 0.1 * i) for i in range(10)]) - path.path_cost

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

    return model_params


def generate_feasible_test_case(
    num_paths=2,
    num_periods=3,
    num_prices=3,
    uncertainty_dim=2,
    base_demand_range=(500.0, 10000.0),
    price_range=(1000.0, 2100.0),
    base_price_ratio=0.5,
    cost_ratio=0.8,
    demand_sensitivity=1.0,
    uncertainty_std_ratio=0.3,
    seed=42
):
    """
    生成一个可行的测试用例，用于测试 SOCP4LDR 算法。

    参数:
    ----------
    num_paths : int
        路径数量 (|Φ|).
    num_periods : int
        时间段数量 (|T|).
    num_prices : int
        价格点数量.
    uncertainty_dim : int
        不确定性维度 (I1).
    base_demand_range : tuple (min, max)
        基础需求 d0 的取值范围.
    price_range : tuple (min, max)
        价格点的取值范围.
    base_price_ratio : float
        基准价格 p_hat 相对于最低价格的比例.
    cost_ratio : float
        成本相对于价格的比例 (c = p * cost_ratio).
    demand_sensitivity : float
        需求对价格的敏感度 (a).
    uncertainty_std_ratio : float
        不确定性标准差相对于均值的比例.
    safety_factor : float
        为初始容量 Y 添加的安全余量 (Y = Y_min * safety_factor).
    seed : int, optional
        随机种子，用于结果复现.

    返回:
    -------
    dict
        包含所有必要参数的字典，可直接用于 SOCP4LDR 类。
    """
    if seed is not None:
        np.random.seed(seed)

    # --- 1. 定义基础集合 ---
    phi_list = [f"P{i+1}" for i in range(num_paths)]
    t_list = list(range(1, num_periods + 1))
    p_list = np.linspace(price_range[0], price_range[1], num_prices).tolist()
    p_min = min(p_list)

    # --- 2. 生成路径相关参数 ---
    p_hat = {}
    t_d_phi = {}
    for phi in phi_list:
        # 基准价格
        p_hat[phi] = p_min * base_price_ratio
        # 需求有效期
        t_d_phi[phi] = np.random.randint(max(1, num_periods - 1), num_periods)

    # --- 3. 生成成本系数 ---
    c_phi_tp = {}
    for phi in phi_list:
        for t in t_list:
            for p in p_list:
                c_phi_tp[(phi, t, p)] = p * cost_ratio

    # --- 4. 生成基础需求 d0 ---
    d_0_phi_t = {}
    for phi in phi_list:
        for t in t_list:
            if t <= t_d_phi[phi]:
                d_0_phi_t[(phi, t)] = np.random.uniform(base_demand_range[0], base_demand_range[1])
            else:
                d_0_phi_t[(phi, t)] = 0.0

    # --- 6. 安全地设置不确定性参数 ---
    # 均值设为0，这样最坏情况下的期望需求就是 d0
    mu = np.zeros(uncertainty_dim)
    # 方差设置为较小的正值
    sigma_sq = np.array([ (base_demand_range[1] * uncertainty_std_ratio) ** 2 for _ in range(uncertainty_dim) ])
    # 协方差矩阵 (无相关性)
    Sigma = np.diag(sigma_sq)

    # 关键：让不确定性降低需求 (d_z 为负)
    # 这样最坏情况 (z 为负) 会增加需求，但增量可控
    d_z_phi_t_i = {}
    for phi in phi_list:
        for t in t_list:
            for i in range(uncertainty_dim):
                d_z_phi_t_i[(phi, t, i)] = -0.5 * (base_demand_range[1] / uncertainty_dim)

    # --- 7. 构造网络参数 (占位) ---
    # 假设每条路径占用一个唯一的边
    paths = {}
    A_prime = {}
    for i, phi in enumerate(phi_list):
        edge = (f"N{i}", f"N{i+1}")
        paths[phi] = [edge]
        A_prime[edge] = 5000.0

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
    }

    print("✅ 成功生成可行测试用例!")
    print(f"  路径数: {num_paths}, 时段数: {num_periods}, 价格点: {num_prices}, 不确定性维度: {uncertainty_dim}")
    return test_case
