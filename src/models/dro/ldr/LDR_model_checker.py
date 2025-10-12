import numpy as np

def validate_cone_constraints(solution, I1, pi_solution):
    """
    测试 1: 验证对偶变量 pi_q 是否满足二阶锥 (SOC) 约束，并统计紧/非紧约束数量。
    锥约束形式: || [π_q[3i], π_q[3i+1]] ||_2 ≤ π_q[3i+2], 且 π_q[3i+2] ≥ 0

    参数:
    - solution: 模型解的字典
    - I1: 不确定性维度
    - pi_solution: 从模型中提取的 pi_q 解，格式为 {q: [pi_vals]}
    """
    print("🧪 测试 1: 验证锥约束")
    all_valid = True
    total_active = 0
    total_inactive = 0
    tolerance = 1e-6
    activity_tolerance = 1e-4  # 用于判断是否为紧约束的容差

    for q, pi_vals in pi_solution.items():
        valid = True
        active_count = 0
        inactive_count = 0

        # --- 检查每个 (3i, 3i+1, 3i+2) 块 ---
        for i in range(I1):
            x, y, t_val = pi_vals[3*i], pi_vals[3*i+1], pi_vals[3*i+2]
            norm_val = (x**2 + y**2)**0.5

            # 1. 验证约束是否满足
            if t_val < -tolerance or norm_val > t_val + tolerance:
                valid = False
                print(f"  ❌ 锥约束违反 for q={q}, block {i}: ||[x,y]||={norm_val:.6f} > t={t_val:.6f}")
            else:
                # 2. 判断是紧约束还是非紧约束
                if abs(norm_val - t_val) <= activity_tolerance:
                    active_count += 1
                else:
                    inactive_count += 1

        # --- 检查聚合块 (3*I1, 3*I1+1, 3*I1+2) ---
        agg_x, agg_y, agg_t = pi_vals[3*I1], pi_vals[3*I1+1], pi_vals[3*I1+2]
        agg_norm_val = (agg_x**2 + agg_y**2)**0.5

        if agg_t < -tolerance or agg_norm_val > agg_t + tolerance:
            valid = False
            print(f"  ❌ 聚合锥约束违反 for q={q}: ||[x,y]||={agg_norm_val:.6f} > t={agg_t:.6f}")
        else:
            if abs(agg_norm_val - agg_t) <= activity_tolerance:
                active_count += 1
            else:
                inactive_count += 1

        # --- 输出每个 q 的结果 ---
        if valid:
            print(f"  ✅ q={q} 的所有锥约束均满足。 (紧: {active_count}, 非紧: {inactive_count})")
            total_active += active_count
            total_inactive += inactive_count
        else:
            all_valid = False

    # --- 输出总计 ---
    print(f"  📊 总计: 紧约束 {total_active} 个, 非紧约束 {total_inactive} 个.")
    if all_valid:
        print("  🎉 测试 1 通过！所有对偶变量均满足锥约束。\n")
    else:
        print("  ⚠️ 测试 1 失败！存在对偶变量不满足锥约束。\n")

    return all_valid

def validate_delta_and_R_constraints(solution, model_params):
    """
    测试 2: 验证原始约束 (Δ=0 和 R=0) 是否被满足。

    参数:
    - solution: 模型解的字典
    - model_params: 一个包含模型参数的字典，例如:
        {
            'phi_list': [...],
            't_list': [...],
            't_d_phi': {...},
            'd_0_phi_t': {...},
            'd_z_phi_t_i': {...},
            'p_hat': {...},
            'a': ...,
            'p_list': [...]
        }
    """
    print("🧪 测试 2: 验证 Δ 和 R 的边界约束 (KKT 条件 - 原始可行性)")
    phi_list = model_params['phi_list']
    t_list = model_params['t_list']
    t_d_phi = model_params['t_d_phi']
    d_0_phi_t = model_params['d_0_phi_t']
    d_z_phi_t_i = model_params['d_z_phi_t_i']
    p_hat = model_params['p_hat']
    a = model_params['a']
    p_list = model_params['p_list']
    I1 = len(solution['s'])
    Y_value = solution['Y']

    all_valid = True
    tolerance = 1e-3

    for phi in phi_list:
        for t in t_list:
            # 1. 计算 Δ0, Δz, Δu
            # Δ0 = R0 - Y + sum_{t'<t} (d0 - a*sum(p*G0) - a*p_hat)
            delta0_val = solution['R0'][(phi, t)] - Y_value[phi]
            for tp in t_list:
                if tp < t:
                    sum_p_G0 = sum(solution['G0'][(phi, tp, p)] * p for p in p_list)
                    demand_tp = d_0_phi_t.get((phi, tp), 0.0)
                    delta0_val += demand_tp - a * sum_p_G0 - a * p_hat[phi]

            # Δz_i = Rz_i + sum_{t'<t} (d_z_i - a*sum(p*Gz_i))
            delta_z_vals = []
            for i in range(I1):
                delta_z_i = solution['Rz'][(phi, t, i)]
                for tp in t_list:
                    if tp < t:
                        sum_p_Gz = sum(solution['Gz'][(phi, tp, p, i)] * p for p in p_list)
                        dz_val = d_z_phi_t_i.get((phi, tp, i), 0.0)
                        delta_z_i += dz_val - a * sum_p_Gz
                delta_z_vals.append(delta_z_i)

            # Δu_k = Ru_k - a * sum_{t'<t} sum_p (p * Gu_k)
            delta_u_vals = []
            for k in range(I1):
                delta_u_k = solution['Ru'][(phi, t, k)]
                for tp in t_list:
                    if tp < t:
                        sum_p_Gu = sum(solution['Gu'][(phi, tp, p, k)] * p for p in p_list)
                        delta_u_k -= a * sum_p_Gu
                delta_u_vals.append(delta_u_k)

            # 2. 根据 t_d_phi 检查约束
            t_deadline = t_d_phi.get(phi, 0)
            if 1 <= t <= t_deadline:
                # 应该满足 Δ = 0
                if abs(delta0_val) > tolerance:
                    all_valid = False
                    print(f"  ❌ Δ0 约束违反 Δ = 0  for ({phi}, {t}): {delta0_val:.6f} ")
                for i, dz in enumerate(delta_z_vals):
                    if abs(dz) > tolerance:
                        all_valid = False
                        print(f"  ❌ Δz 约束违反Δ = 0 for ({phi}, {t}, i={i}): {dz:.6f}")
                for k, du in enumerate(delta_u_vals):
                    if abs(du) > tolerance:
                        all_valid = False
                        print(f"  ❌ Δu 约束违反Δ = 0 for ({phi}, {t}, k={k}): {du:.6f}")
            else:
                # 应该满足 R = 0
                if abs(solution['R0'][(phi, t)]) > tolerance:
                    all_valid = False
                    print(f"  ❌ R0 约束违反 R = 0 for ({phi}, {t}): {solution['R0'][(phi, t)]:.6f}")
                for i in range(I1):
                    if abs(solution['Rz'][(phi, t, i)]) > tolerance:
                        all_valid = False
                        print(f"  ❌ Rz 约束违反 R = 0 for ({phi}, {t}, i={i}): {solution['Rz'][(phi, t, i)]:.6f}")
                for k in range(I1):
                    if abs(solution['Ru'][(phi, t, k)]) > tolerance:
                        all_valid = False
                        print(f"  ❌ Ru 约束违反 R = 0 for ({phi}, {t}, k={k}): {solution['Ru'][(phi, t, k)]:.6f}")

    if all_valid:
        print("  🎉 测试 2 通过！所有 Δ 和 R 的边界约束均满足。\n")
    else:
        print("  ⚠️ 测试 2 失败！存在原始约束违反。\n")
    return all_valid


def validate_affine_constraints_tightness(solution, pi_solution, model_params):
    """
    验证仿射约束在最坏情况下是否被满足（通过检查 h^T π_q + alpha0_q <= 0）。
    这是 LDR 框架下验证原始可行性的正确方式。
    """
    print("🧪 测试: 验证仿射约束的可行性 (通过对偶)")
    all_valid = True
    tolerance = 1e-6

    for q, pi_vals in pi_solution.items():
        # 计算 h^T π_q
        h_dot_pi = sum(h_val * pi_val for h_val, pi_val in zip(model_params['h'], pi_vals))
        # 从 solution 中获取 alpha0_q (这需要您在 get_solution 中也返回 alpha0)
        alpha0_q = solution['alpha'][q]['alpha0']
        # 检查 h^T π_q <= -alpha0_q
        if h_dot_pi > -alpha0_q + tolerance:
            all_valid = False
            print(f"  ❌ 仿射约束违反 for q={q}: h^T π_q ({h_dot_pi:.6f}) > -alpha0_q ({-alpha0_q:.6f})")
        else:
            print(f"  ✅ 仿射约束满足 for q={q}")

    if all_valid:
        print("  🎉 所有仿射约束均满足！\n")
    else:
        print("  ⚠️ 部分仿射约束违反！\n")
    return all_valid

def validate_variable_bounds(solution, model_params):
    """
    测试 3: 验证变量边界和业务逻辑。

    参数:
    - solution: 模型解的字典
    - model_params: 包含 'phi_list', 't_list', 'p_list' 的字典
    """
    print("🧪 测试 3: 验证变量边界和业务逻辑")
    phi_list = model_params['phi_list']
    t_list = model_params['t_list']
    p_list = model_params['p_list']
    I1 = len(solution['s'])
    all_valid = True
    tolerance = 1e-6

    # 1. 检查 G0: 应在 [0, 1] 且 sum_p G0 ≈ 1
    for phi in phi_list:
        for t in t_list:
            sum_G0 = sum(solution['G0'][(phi, t, p)] for p in p_list)
            if abs(sum_G0 - 1.0) > 1e-4:  # 允许稍大一点的容差
                all_valid = False
                print(f"  ❌ G0 求和不为 1 for ({phi}, {t}): {sum_G0:.6f}")
            for p in p_list:
                G0_val = solution['G0'][(phi, t, p)]
                if G0_val < -tolerance or G0_val > 1.0 + tolerance:
                    all_valid = False
                    print(f"  ❌ G0 超出 [0,1] 范围 for ({phi}, {t}, {p}): {G0_val:.6f}")

    # 2. 检查 R0: 应为非负
    for phi in phi_list:
        for t in t_list:
            R0_val = solution['R0'][(phi, t)]
            if R0_val < -tolerance:
                all_valid = False
                print(f"  ❌ R0 为负 for ({phi}, {t}): {R0_val:.6f}")

    # 3. 检查 t 和 l: 应 <= 0
    for k, t_val in enumerate(solution['t']):
        if t_val > tolerance:
            all_valid = False
            print(f"  ❌ t[{k}] 应 <= 0: {t_val:.6f}")
    if solution['l'] > tolerance:
        all_valid = False
        print(f"  ❌ l 应 <= 0: {solution['l']:.6f}")

    if all_valid:
        print("  🎉 测试 3 通过！所有变量均满足边界和业务逻辑。\n")
    else:
        print("  ⚠️ 测试 3 失败！存在变量违反边界或逻辑。\n")
    return all_valid

def validate_objective_value(solution, model_params):
    """
    测试 4: 验证目标函数值的合理性（与确定性基准比较）。

    参数:
    - solution: 模型解的字典
    - model_params: 包含完整模型参数的字典
    """
    print("🧪 测试 4: 验证目标函数值的合理性")
    # 计算确定性基准：在 z = mu 时的收益
    phi_list = model_params['phi_list']
    t_list = model_params['t_list']
    p_list = model_params['p_list']
    c_phi_tp = model_params['c_phi_tp']
    d_0_phi_t = model_params['d_0_phi_t']
    d_z_phi_t_i = model_params['d_z_phi_t_i']
    mu = model_params['mu']
    a = model_params['a']
    p_hat = model_params['p_hat']

    deterministic_revenue = 0.0
    for phi in phi_list:
        for t in t_list:
            for p in p_list:
                # 计算在 z=mu 时的需求
                demand_z = sum(d_z_phi_t_i.get((phi, t, i), 0.0) * mu[i] for i in range(len(mu)))
                total_demand = d_0_phi_t.get((phi, t), 0.0) + demand_z
                # 计算实际销售量 (不能超过需求和容量，此处简化)
                # Todo
                G0_val = solution['G0'][(phi, t, p)]
                sales = G0_val * min(total_demand, 1e6) # 简化处理
                deterministic_revenue += c_phi_tp[(phi, t, p)] * G0_val

    # DRO 目标值 (beta) 应 <= 确定性基准
    beta_val = solution.get('obj_val', None)
    if beta_val is not None:
        if beta_val <= deterministic_revenue + 1e-4:
            print(f"  ✅ DRO 目标值 ({beta_val:.4f}) <= 确定性基准 ({deterministic_revenue:.4f})，符合预期。")
            return True
        else:
            print(f"  ❌ DRO 目标值 ({beta_val:.4f}) > 确定性基准 ({deterministic_revenue:.4f})，不符合预期！")
            return False
    else:
        print("  ⚠️ 未提供 obj_val，跳过此测试。")
        return True

def run_all_validations(solution, model_params, pi_solution=None):
    """
    运行所有验证测试。

    参数:
    - solution: 模型解字典
    - model_params: 模型参数字典
    - pi_solution: (可选) 对偶变量 pi_q 的解
    """
    print("=" * 60)
    print("🔍 开始执行模型解的全面验证")
    print("=" * 60)

    results = []
    # # 测试 3 和 4 只需要 solution 和 model_params
    # results.append(validate_variable_bounds(solution, model_params))
    # results.append(validate_objective_value(solution, model_params))
    results.append(validate_affine_constraints_tightness(solution, pi_solution, model_params))

    # 测试 2 需要完整的 model_params
    results.append(validate_delta_and_R_constraints(solution, model_params))

    # 测试 1 需要 pi_solution
    if pi_solution is not None:
        results.append(validate_cone_constraints(solution, len(solution['s']), pi_solution))
    else:
        print("🧪 测试 1: 跳过 (未提供 pi_solution)\n")
        results.append(True) # 或 False，取决于您是否认为这是必需的

    print("=" * 60)
    if all(results):
        print("🎉 所有验证测试均已通过！模型解在数学和逻辑上是可靠的。")
    else:
        print("⚠️ 部分验证测试失败。请根据上述详细信息检查模型和数据。")
    print("=" * 60)

# ================= 使用示例 =================
if __name__ == "__main__":
    # 假设您已经有了 solution, pi_solution, 和 model_params
    # model_params = {
    #     'I1': 2,
    #     'phi_list': ['P1', 'P2'],
    #     't_list': [1, 2, 3],
    #     'p_list': [10.0, 15.0],
    #     't_d_phi': {'P1': 2, 'P2': 3},
    #     'd_0_phi_t': {('P1', 1): 8.0, ...},
    #     'd_z_phi_t_i': {('P1', 1, 0): 2.0, ...},
    #     'p_hat': {'P1': 5.0, 'P2': 6.0},
    #     'a': 1.0,
    #     'c_phi_tp': {('P1', 1, 10.0): 2.0, ...},
    #     'mu': [1.0, 0.4]
    # }
    #
    # run_all_validations(solution, model_params, pi_solution)
    pass
