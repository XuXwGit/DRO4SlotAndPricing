import math
from random import random
from typing import Any, Dict

import numpy as np

from src.config.config import Config
from src.utils.data_manager import DataManager


def construct_model_params(data_manager: 'DataManager', 
                           price_sensitivity=0.5,
                           uncertainty_dim=1, 
                           uncertainty_std_ratio=0.01) -> Dict[str, Any]:
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
        num_periods = data_manager.time_horizon
        t_list = list(range(0, num_periods + 1))

        # # 3. 提取价格点 (p_list)
        # p_list = {}
        # for request in data_manager.requests:
        #     # 假设 Request 类有一个 price_set 属性，它是一个列表
        #     for path in request.laden_path_set.values():
        #         p_list[path.id] = getattr(request, 'price_set', [math.ceil(request.long_haul_price * (1.5 + 0.1 * i) - path.path_cost) for i in range(10)])

        # # 4. 提取基准价格 p_hat
        # p_hat = {}
        # for request in data_manager.requests:
        #     for path in request.laden_path_set.values():
        #         p_hat[path.id] = request.long_haul_price

        # --- 2. 价格参数 ---
        p_hat = {}
        t_d_phi = {}
        max_demand = 0.0
        price_range = [10, 100]
        num_prices = 10
        for phi in phi_list:
            p_hat[phi] = price_range[0] * 0.5
            t_d_phi[phi] = np.random.randint(max(1, num_periods - 1), num_periods)
        p_list = np.linspace(price_range[0], price_range[1], num_prices).tolist()


        # 7. 提取价格敏感度 a
        a = price_sensitivity

        # 6. 提取需求有效期 t_d_phi
        t_d_phi = {}
        for cp in data_manager.container_paths:
            t_d_phi[cp.container_path_id] = getattr(cp, 'origin_time', data_manager.time_horizon)

        # 5. 提取基础需求 d_0_phi_t
        d_0_phi_t = {}
        for request in data_manager.requests:
            d0 = getattr(request, 'mean_demand', 10.0) + getattr(request, 'variance_demand', 10.0)
            for cp in request.laden_path_set.values():
                phi = cp.container_path_id
                for t in t_list:
                    if t < t_d_phi[phi]:
                        d_0_phi_t[(phi, t)] = d0
                    else:
                        d_0_phi_t[(phi, t)] = 0

        # 8. 提取 per-period  revenue:  c_phi_tp
        # c_φtp = p Kφt = p dφt − a(p − pˆφ) = p dφt − a p^2 + a p pˆφ
        c_phi_tp = {}
        for request in data_manager.requests:
            for cp in request.laden_path_set.values():
                for t in t_list:
                    for p in p_list:
                        if t < t_d_phi[cp.container_path_id]:
                            c_phi_tp[(cp.container_path_id, t, p)] = max(0, p * d_0_phi_t[(cp.container_path_id, t)] - a * p * p + a * p * p_hat[cp.container_path_id])
                        else:
                            c_phi_tp[(cp.container_path_id, t, p)] = 0

        # 9. 提取需求对 z 的敏感度 d_z_phi_t_i
        # 不确定性维度 I1 设为港口数量
        I1 = uncertainty_dim
        d_z_phi_t_i = {}
        for phi in phi_list:
            for t in t_list:
                for i in range(uncertainty_dim):
                    if t < t_d_phi[phi]:
                        # 波动与基础需求成比例
                        d_z_phi_t_i[(phi, t, i)] = math.floor(0.02 * d_0_phi_t[(phi, t)])
                    else:
                        d_z_phi_t_i[(phi, t, i)] = 0


        # 10. 提取不确定性参数 (mu, sigma_sq, Sigma)
        mu = np.zeros(uncertainty_dim)
        # mu= 1 *np.ones(uncertainty_dim)
        # 方差 = (最大需求 × std_ratio)^2
        sigma_sq = np.array([
            (uncertainty_std_ratio) ** 2
            for _ in range(uncertainty_dim)
        ])
        # sigma_sq = 0.05*np.ones(uncertainty_dim)
        Sigma = np.diag(sigma_sq)

        # 11. 构建 model_params 字典
        model_params = {
            'num_paths': len(phi_list),
            'num_periods': num_periods,
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
            "cost_cov": np.sum(Sigma),  # 添加 cost_cov
            "paths": paths,
            "A_prime": A_prime,
        }
    except Exception as e:
        print(f"Error constructing model parameters: {e}")
        raise e

    print_params_info(model_params)
    # - A_prime: 弧容量字典
    return model_params


def generate_feasible_test_case(
    num_paths=1,
    paths = None,
    deadline=None,
    num_periods=2,
    num_prices=5,
    uncertainty_dim=1,
    base_mean_demand =30,
    base_var_demand = 10, 
    price_range=(10.0, 100.0),
    demand_sensitivity=0.5,
    uncertainty_std_ratio=0.1,
    base_capacity=1000,
    seed=42
):
    if seed is not None:
        np.random.seed(seed)

    # --- 1. 基础集合 ---
    if paths is None:
        phi_list = [f"P{i+1}" for i in range(num_paths)]
    else:
        phi_list = paths.keys()
    t_list = list(range(0, num_periods + 1))

    # --- 2. 路径参数 ---
    if paths is None:
        paths = {}
        for i, phi in enumerate(phi_list):
                paths[phi] = []
                edge = (f"N{i}", f"N{i+1}")
                paths[phi] = [edge]
                paths[phi].append((f"N{i+1}", f"N{i+2}"))
                paths[phi].append((f"N{i+2}", f"N{i+3}"))
                if random() < 0.5:
                    paths[phi].append((f"N{i+3}", f"N{i+4}"))
                    end = f"N{i+4}"
                else:
                    paths[phi].append((f"N{i+3}", f"N{i+5}"))
                    end = f"N{i+4}"
                if end != f"N{i * 2}":
                    paths[phi].append((end, f"N{i * 2}"))

    # --- 4. 动态设置容量 (确保容量 > 最大需求) ---
    A_prime = {}
    for phi, arcs in paths.items():
        for arc in arcs:
            A_prime[arc] = base_capacity

    # --- 2. 价格参数 ---
    p_hat = {}
    t_d_phi = {}
    max_demand = 0.0
    for phi in phi_list:
        p_hat[phi] = price_range[0] * Config.LONG_HAUL_COEFFICIENT
        t_d_phi[phi] = np.random.randint(max(1, num_periods - 1), num_periods)
    p_list = np.linspace(price_range[0], price_range[1], num_prices).tolist()

    # --- 3. DDL ---
    if deadline is None:
        t_d_phi = {}
        for phi in phi_list:
            t_d_phi[phi] = np.random.randint(max(1, num_periods - 1), num_periods)
    else:
        t_d_phi = deadline

    # --- 3. 基础需求---
    d_0_phi_t = {}
    for phi in phi_list:
        for t in t_list:
            if t <= t_d_phi[phi]:
                d_val = math.floor(max(np.random.normal(base_mean_demand, base_var_demand), 0))
                d_0_phi_t[(phi, t)] = d_val
                max_demand = max(max_demand, d_val)
            else:
                d_0_phi_t[(phi, t)] = 0.0

    # --- 5. 收益系数 c_phi_t_p = p K_phi_t = p[d_phi_t - a(p - p_hat)] ---
    c_phi_t_p = {}
    for phi in phi_list:
        for t in t_list:
            for p in p_list:
                if t <= t_d_phi[phi]:
                    c_phi_t_p[(phi, t, p)] = max(p * (d_0_phi_t[(phi, t)] - demand_sensitivity * (p - p_hat[phi])), 0)
                else:
                    c_phi_t_p[(phi, t, p)] = 0.0

    # --- 6. 不确定性参数 (减小方差) ---
    mu = np.zeros(uncertainty_dim)
    # mu= 1 *np.ones(uncertainty_dim)
    # 方差 = (最大需求 × std_ratio)^2
    sigma_sq = np.array([
        (uncertainty_std_ratio) ** 2
        for _ in range(uncertainty_dim)
    ])
    # sigma_sq = 0.05*np.ones(uncertainty_dim)
    Sigma = np.diag(sigma_sq)

    # --- 7. 需求波动 ---
    d_z_phi_t_i = {}
    for phi in phi_list:
        for t in t_list:
            for i in range(uncertainty_dim):
                # 波动与基础需求成比例
                d_z_phi_t_i[(phi, t, i)] = math.floor(0.02 * d_0_phi_t[(phi, t)])

    # --- 8. 返回结果 ---
    test_model_params = {
        'num_paths': len(phi_list),
        'num_periods': num_periods,
        "I1": uncertainty_dim,
        "phi_list": phi_list,
        "t_list": t_list,
        "p_list": p_list,
        "p_hat": p_hat,
        "c_phi_tp": c_phi_t_p,
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

    print_params_info(test_model_params)
    return test_model_params


def print_params_info(model_params):
    print(f"  路径数: {model_params['num_paths']}, 航段数: {len(model_params['A_prime'])}, 时段数: {model_params['num_periods']}")
    print(f"  不确定性维度: {model_params['I1']}")
    print(f"  基础需求范围: {min(model_params['d_0_phi_t'].values())} - {max(model_params['d_0_phi_t'].values())}")
    print(f"  价格范围: {min(model_params['p_list'])} - {max(model_params['p_list'])}")
    print(f"  容量范围: {min(model_params['A_prime'].values())} - {max(model_params['A_prime'].values())}")
    print(f"  价格弹性: {model_params['a']}")
    print(f"  方差 sigma_sq: {model_params['sigma_sq'][0]:.2f}")