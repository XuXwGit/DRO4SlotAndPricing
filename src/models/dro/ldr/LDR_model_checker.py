import numpy as np

def validate_capacity_constraints(solution, model_params):
    """
    测试 2: 验证容量约束
    每条边 e 的容量约束: ∑_{i=1}^{I} (X_{i,e} + Y_{i,e}) ≤ A'_e
    """
    print("🧪 测试 3: 验证容量约束")
    all_valid = True
    total_active = 0
    tolerance = 1e-6
    activity_tolerance = 1e-4
    for edge, capacity in model_params['A_prime'].items():
        valid = True
        active_count = 0
        inactive_count = 0
        expr_value = 0
        for phi in model_params['phi_list']:
            if edge in model_params['paths'].get(phi, []):
                expr_value = solution['X'][phi] + solution['Y'][phi]

        if expr_value > capacity + tolerance:
                    valid = False
                    print(f"  ❌ 容量约束违反 for edge={edge}: {expr_value:.6f} > {capacity:.6f}")
        else:
                    # 判断是否为紧约束
                    if abs(expr_value - capacity) <= activity_tolerance:
                        active_count += 1
                    else:
                        inactive_count += 1

        if valid:
            print(f"  ✅ edge={edge} 的所有容量约束均满足。 (紧: {active_count}, 非紧: {inactive_count})")
            total_active += active_count

def validate_cone_constraints(solution, I1):
    """
    测试 1: 验证对偶变量 pi_q 是否满足二阶锥 (SOC) 约束。
    锥结构: || [π_q[3i+1], π_q[3i+2]] ||_2 ≤ π_q[3i] 且 π_q[3i] ≥ 0
    共 I1+1 个锥（I1 个分量锥 + 1 个聚合锥）
    """
    print("🧪 测试 1: 验证锥约束")
    all_valid = True
    total_active = 0
    total_inactive = 0
    tolerance = 1e-6
    activity_tolerance = 1e-4  # 判断紧约束的容差

    for q, pi_vals in solution['pi'].items():
        valid = True
        active_count = 0
        inactive_count = 0

        # 遍历 I1+1 个锥（含聚合锥）
        for i in range(I1 + 1):
            if i < I1:
                t_val = pi_vals[3 * i]
                y1 = pi_vals[3 * i + 1]
                y2 = pi_vals[3 * i + 2]
                block_desc = f"分量锥 {i}"
            else:
                t_val = pi_vals[3 * I1]
                y1 = pi_vals[3 * I1 + 1]
                y2 = pi_vals[3 * I1 + 2]
                block_desc = "聚合锥"

            norm_val = np.sqrt(y1**2 + y2**2)

            # 检查锥约束：t >= ||y|| 且 t >= 0
            if t_val < -tolerance or norm_val > t_val + tolerance:
                valid = False
                print(f"  ❌ 锥约束违反 for q={q}, {block_desc}: ||y||={norm_val:.6f} > t={t_val:.6f} (或 t<0)")
            else:
                # 判断是否为紧约束
                if abs(norm_val - t_val) <= activity_tolerance:
                    active_count += 1
                else:
                    inactive_count += 1

        if valid:
            print(f"  ✅ q={q} 的所有锥约束均满足。 (紧: {active_count}, 非紧: {inactive_count})")
            total_active += active_count
            total_inactive += inactive_count
        else:
            all_valid = False

    print(f"  📊 总计: 紧约束 {total_active} 个, 非紧约束 {total_inactive} 个.")
    if all_valid:
        print("  🎉 测试 1 通过！所有对偶变量均满足锥约束。\n")
    else:
        print("  ⚠️ 测试 1 失败！存在对偶变量不满足锥约束。\n")

    return all_valid


def validate_delta_and_R_constraints(solution, model_params):
    """
    测试 2: 验证原始约束 (Δ=0 和 R=0) 是否被满足。
    """
    print("🧪 测试 2: 验证 Δ 和 R 的边界约束 (Δ=0 和 R=0)")
    phi_list = model_params['phi_list']
    t_list = model_params['t_list']
    t_d_phi = model_params.get('t_d_phi', {})
    d_0_phi_t = model_params.get('d_0_phi_t', {})
    d_z_phi_t_i = model_params.get('d_z_phi_t_i', {})
    p_hat = model_params.get('p_hat', {})
    a = model_params['a']
    p_list = model_params['p_list']
    I1 = len(solution['s'])
    Y_sol = solution['Y']

    all_valid = True
    tolerance = 1e-4  # 稍宽松，因涉及多步累加

    for phi in phi_list:
        t_deadline = t_d_phi.get(phi, 0)
        for t in t_list:
            if t < 1:
                continue

            # --- 计算 Δ0 ---
            delta0_val = solution['R0'][(phi, t)] - Y_sol[phi]
            for tp in t_list:
                if tp < t:
                    sum_p_G0 = sum(solution['G0'][(phi, tp, p)] * p for p in p_list)
                    demand_tp = d_0_phi_t.get((phi, tp), 0.0)
                    delta0_val += demand_tp + a * p_hat.get(phi, 0.0) - a * sum_p_G0

            # --- 计算 Δz_i ---
            delta_z_vals = []
            for i in range(I1):
                delta_z_i = solution['Rz'][(phi, t, i)]
                for tp in t_list:
                    if tp < t:
                        dz_val = d_z_phi_t_i.get((phi, tp, i), 0.0)
                        sum_p_Gz = sum(solution['Gz'][(phi, tp, p, i)] * p for p in p_list)
                        delta_z_i += dz_val - a * sum_p_Gz
                delta_z_vals.append(delta_z_i)

            # --- 计算 Δu_k ---
            delta_u_vals = []
            for k in range(I1):
                delta_u_k = solution['Ru'][(phi, t, k)]
                for tp in t_list:
                    if tp < t:
                        sum_p_Gu = sum(solution['Gu'][(phi, tp, p, k)] * p for p in p_list)
                        delta_u_k -= a * sum_p_Gu
                delta_u_vals.append(delta_u_k)

            # --- 根据时间区间验证约束 ---
            if 1 <= t <= t_deadline:
                # 需满足 Δ = 0
                if abs(delta0_val) > tolerance:
                    all_valid = False
                    print(f"  ❌ Δ0 ≠ 0 for ({phi}, {t}): {delta0_val:.6f}")
                for i, dz in enumerate(delta_z_vals):
                    if abs(dz) > tolerance:
                        all_valid = False
                        print(f"  ❌ Δz[{i}] ≠ 0 for ({phi}, {t}): {dz:.6f}")
                for k, du in enumerate(delta_u_vals):
                    if abs(du) > tolerance:
                        all_valid = False
                        print(f"  ❌ Δu[{k}] ≠ 0 for ({phi}, {t}): {du:.6f}")
            elif t > t_deadline:
                # 需满足 R = 0
                if abs(solution['R0'][(phi, t)]) > tolerance:
                    all_valid = False
                    print(f"  ❌ R0 ≠ 0 for ({phi}, {t}): {solution['R0'][(phi, t)]:.6f}")
                for i in range(I1):
                    if abs(solution['Rz'][(phi, t, i)]) > tolerance:
                        all_valid = False
                        print(f"  ❌ Rz[{i}] ≠ 0 for ({phi}, {t}): {solution['Rz'][(phi, t, i)]:.6f}")
                for k in range(I1):
                    if abs(solution['Ru'][(phi, t, k)]) > tolerance:
                        all_valid = False
                        print(f"  ❌ Ru[{k}] ≠ 0 for ({phi}, {t}): {solution['Ru'][(phi, t, k)]:.6f}")

    if all_valid:
        print("  🎉 测试 2 通过！所有 Δ 和 R 的边界约束均满足。\n")
    else:
        print("  ⚠️ 测试 2 失败！存在原始约束违反。\n")
    return all_valid


def validate_support_duality_constraints_tightness(solution, model_params, tolerance: float = 1e-6):
    """
    测试 3: 验证对偶线性约束是否满足。
    """
    print("🧪 测试 3: 验证对偶支撑函数约束")
    C = model_params['C']
    D = model_params['D']
    d_vec = model_params['d']
    h_vec = model_params['h']

    # 鲁棒获取 I1
    I1 = len(solution['s'])  # 或 len(model_params['mu'])
    dim_pi = len(h_vec)

    all_valid = True

    for q, pi_vals in solution['pi'].items():
        print(f"\n🔍 检查 q = {q}")

        # (1) C^T π_q == alpha_z_q
        print(f"C^T π_q == alpha_z_q")
        C_dot_pi = [sum(C[j][i] * pi_vals[j] for j in range(dim_pi)) for i in range(I1)]
        alpha_z_q = solution['alpha'][q]['alpha_z']
        for i in range(I1):
            if abs(C_dot_pi[i] - alpha_z_q[i]) > tolerance:
                all_valid = False
                print(f"  ❌ C^T π_q[{i}] = {C_dot_pi[i]:.6f} != alpha_z_q[{i}] = {alpha_z_q[i]:.6f}")
            else:
                print(f"  ✅ C^T π_q[{i}] = alpha_z_q[{i}]")
        print(f"\n")

        # (2) D^T π_q == alpha_u_q
        print(f"D^T π_q == alpha_u_q")
        D_dot_pi = [sum(D[j][k] * pi_vals[j] for j in range(dim_pi)) for k in range(I1)]
        alpha_u_q = solution['alpha'][q]['alpha_u']
        for k in range(I1):
            if abs(D_dot_pi[k] - alpha_u_q[k]) > tolerance:
                all_valid = False
                print(f"  ❌ D^T π_q[{k}] = {D_dot_pi[k]:.6f} != alpha_u_q[{k}] = {alpha_u_q[k]:.6f}")
            else:
                print(f"  ✅ D^T π_q[{k}] = alpha_u_q[{k}]")
        print(f"\n")

        # (3) d^T π_q == gamma_q
        print(f"d^T π_q == gamma_q")
        d_dot_pi = sum(d_val * pi_val for d_val, pi_val in zip(d_vec, pi_vals))
        gamma_q = solution['alpha'][q]['gamma']
        if abs(d_dot_pi - gamma_q) > tolerance:
            all_valid = False
            print(f"  ❌ d^T π_q = {d_dot_pi:.6f} != gamma_q = {gamma_q:.6f}")
        else:
            print(f"  ✅ d^T π_q = gamma_q")
        print(f"\n")

        # (4) h^T π_q <= -alpha0_q
        print(f"h^T π_q <= -alpha0_q")
        h_dot_pi = sum(h_val * pi_val for h_val, pi_val in zip(h_vec, pi_vals))
        alpha0_q = solution['alpha'][q]['alpha0']
        rhs = -alpha0_q
        if h_dot_pi > rhs + tolerance:
            all_valid = False
            print(f"  ❌ h^T π_q = {h_dot_pi:.6f} > -alpha0_q = {rhs:.6f}")
        else:
            print(f"  ✅ h^T π_q ({h_dot_pi:.6f}) <= -alpha0_q ({rhs:.6f})")
        print(f"\n")

    print(f"\n{'✅ 所有对偶约束满足！' if all_valid else '❌ 存在对偶约束违反！'}")
    return all_valid


def validate_objective_value(solution, model_params, tolerance: float = 1e-6):
    """
    测试 4: 验证目标函数值是否正确。
    """
    print("🧪 测试 4: 验证目标函数值")
    # Stage I
    X = solution['X']
    obj_val_I = sum(p_hat * X[phi] for phi, p_hat in model_params['p_hat'].items())
    # Stage II
    obj_val_II = solution['r']
    obj_val_II += sum(solution['s'][i] * model_params['mu'][i] for i in range(model_params['I1']))
    obj_val_II += sum(solution['t'][k] * model_params['sigma_sq'][k] for k in range(model_params['I1']))
    obj_val_II += solution['l'] * model_params['cost_cov']

    obj_val = obj_val_I + obj_val_II

    if abs(obj_val - solution['obj_val']) > tolerance:
        print(f"  ❌ 目标函数值不匹配: {obj_val:.6f} != {solution['obj_val']:.6f}")
        return False
    else:
        print(f"  ✅ 目标函数值匹配: {obj_val:.6f} = {solution['obj_val']:.6f}")
        return True


def run_all_validations(solution, model_params):
    """
    运行所有验证测试。
    """
    print("=" * 60)
    print("🔍 开始执行模型解的全面验证")
    print("=" * 60)

    I1 = len(solution['s'])
    results = []


    # 测试 2: Δ 和 R 约束
    results.append(validate_delta_and_R_constraints(solution, model_params))

    # 测试 3: 锥约束
    if solution['pi'] is not None:
        results.append(validate_cone_constraints(solution, I1))
    else:
        print("🧪 测试 1: 跳过（未提供 pi 解）\n")
        results.append(True)

    # 测试 4: 对偶线性约束
    results.append(validate_support_duality_constraints_tightness(solution, model_params))

    print("=" * 60)
    if all(results):
        print("🎉 所有验证测试均已通过！模型解在数学和逻辑上是可靠的。")
    else:
        print("⚠️ 部分验证测试失败。请根据上述详细信息检查模型和数据。")
    print("=" * 60)


# ================= 使用示例 =================
if __name__ == "__main__":
    # 示例用法（需替换为实际数据）
    pass
