import numpy as np
from socp import SOCP4LDR

def run_complex_test_case():
    """
    运行一个复杂的测试案例，包含 2 条路径、3 个时间段、2 个价格点、2 维不确定性。
    """
    print("=" * 70)
    print("🧪 运行复杂 SOCP4LDR 测试案例")
    print("=" * 70)

    # --- 1. 定义测试参数 (如上所述) ---
    I1 = 2
    phi_list = ["P1", "P2"]
    t_list = [1, 2, 3]
    p_list = [10.0, 15.0]
    p_hat = {"P1": 5.0, "P2": 6.0}
    c_phi_tp = {
        ("P1", 1, 10.0): 2.0, ("P1", 1, 15.0): 3.0,
        ("P1", 2, 10.0): 2.1, ("P1", 2, 15.0): 3.1,
        ("P1", 3, 10.0): 2.2, ("P1", 3, 15.0): 3.2,
        ("P2", 1, 10.0): 1.8, ("P2", 1, 15.0): 2.8,
        ("P2", 2, 10.0): 1.9, ("P2", 2, 15.0): 2.9,
        ("P2", 3, 10.0): 2.0, ("P2", 3, 15.0): 3.0,
    }
    t_d_phi = {"P1": 2, "P2": 3}

    mu = np.array([1.0, 0.4])
    sigma_sq = np.array([1.0, 0.16])
    Sigma = np.array([[1.0, 0.2], [0.2, 0.16]])

    paths = {
        "P1": [("A", "B"), ("B", "C")],
        "P2": [("A", "D"), ("D", "C")]
    }
    A_prime = {
        ("A", "B"): 100.0,
        ("B", "C"): 80.0,
        ("A", "D"): 90.0,
        ("D", "C"): 85.0
    }

    d_0_phi_t = {
        ("P1", 1): 8.0, ("P1", 2): 7.0, ("P1", 3): 0.0,
        ("P2", 1): 9.0, ("P2", 2): 8.5, ("P2", 3): 8.0
    }
    d_z_phi_t_i = {
        ("P1", 1, 0): 2.0, ("P1", 2, 0): 1.8, ("P1", 3, 0): 0.0,
        ("P1", 1, 1): 0.0, ("P1", 2, 1): 0.0, ("P1", 3, 1): 0.0,
        ("P2", 1, 0): 2.2, ("P2", 2, 0): 2.0, ("P2", 3, 0): 1.8,
        ("P2", 1, 1): 1.0, ("P2", 2, 1): 0.9, ("P2", 3, 1): 0.8,
    }
    a = 1.0

    X_value = {"P1": 2.0, "P2": 3.0}
    Y_value = {"P1": 5.0, "P2": 6.0}

    # --- 2. 初始化模型 ---
    socp = SOCP4LDR(I1=I1, phi_list=phi_list, t_list=t_list, p_list=p_list,
                    p_hat=p_hat, c_phi_tp=c_phi_tp, t_d_phi=t_d_phi)

    socp.set_uncertainty(mu=mu, sigma_sq=sigma_sq, Sigma=Sigma)
    socp.set_network(paths=paths, A_prime=A_prime)
    socp.set_demand_function(d_0_phi_t=d_0_phi_t, a=a, d_z_phi_t_i=d_z_phi_t_i)
    socp.set_Q_list()

    # --- 3. 构建并求解 ---
    socp.build_model()
    socp.set_X_Y_value(X_value=X_value, Y_value=Y_value)

    success, obj_val = socp.solve(verbose=True)

    if success:
        print("\n🎉 复杂案例测试成功！模型已求解。")
        sol = socp.get_solution()
        if sol:
            print(f"\n🔑 关键解摘要:")
            print(f"  最优目标值 β(X,Y) = {obj_val:.6f}")
            print(f"  r = {sol['r']:.6f}")
            print(f"  s = [{sol['s'][0]:.6f}, {sol['s'][1]:.6f}]")
            print(f"  t = [{sol['t'][0]:.6f}, {sol['t'][1]:.6f}]")
            print(f"  l = {sol['l']:.6f}")

            print(f"\n  G0 (概率和应接近 1.0):")
            for phi in phi_list:
                for t in t_list:
                    if t <= t_d_phi[phi]: # 只打印有效期内的
                        total_prob = sum(sol['G0'][(phi, t, p)] for p in p_list)
                        print(f"    G0[{phi},{t},*] sum = {total_prob:.6f}")

            print(f"\n  R0 (应满足 Δ=0 或 R=0 的约束):")
            for phi in phi_list:
                for t in t_list:
                    print(f"    R0[{phi},{t}] = {sol['R0'][(phi, t)]:.6f}")

            # 打印一个具体的 Gz 值作为示例
            example_key = ('P2', 1, 15.0, 1) # P2, t=1, p=15.0, k=1 (z2)
            if example_key in sol['Gz']:
                print(f"\n  示例: Gz{example_key} = {sol['Gz'][example_key]:.6f}")

    else:
        print("\n❌ 复杂案例测试失败！模型未找到最优解。")
        # 尝试计算 IIS 以诊断问题
        try:
            socp.model.computeIIS()
            socp.model.write("complex_model_iis.ilp")
            print("  📄 IIS 已写入 'complex_model_iis.ilp'，请检查以诊断不可行性。")
        except Exception as e:
            print(f"  计算 IIS 时出错: {e}")

if __name__ == "__main__":
    run_complex_test_case()