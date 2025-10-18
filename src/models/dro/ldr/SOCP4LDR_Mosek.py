"""
使用Mosek 建模 重构后的SOCP模型
"""
import logging
from typing import Dict, Any
import os
import sys
import mosek

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_builder import ModelBuilder, timeit_if_debug

class SOCP4LDR_Mosek(ModelBuilder):
    """
    使用 MOSEK Optimizer API (Task) 建模二阶段分布鲁棒优化 (SOCP-based LDR)
    -------------------------------------------------------
    说明：
      - 模型形式如下: max_{X,Y} sum_phi p_hat[phi] * X[phi] + beta_E^LDR(X, Y)
      - 第一阶段决策: X_phi, Y_phi (非负)
      - 第二阶段决策 LDR: G^0, G^{(z)}, G^{(u)}, R^0, R^{(z)}, R^{(u)}
      - alpha/gamma/delta 作为线性表达式 (LinExpr)
      - 利用对偶与锥对偶转换，得到 π_q 等对偶变量，并构造 SOCP 块
    """

    def __init__(self, model_params: Dict[str, Any], debug: bool = False):
        super().__init__(info="SOCP4LDR", solver="mosek")
        self.set_model_params(model_params)

        # 辅助成员变量: 构建变量索引映射（Mosek不支持字典式的访问变量）
        self._var_index = {}
        self._next_var = 0

    def _add_var(self, name: str) -> int:
        idx = self._next_var
        self._var_index[name] = idx
        self._next_var += 1
        return idx

    def _get_var(self, name: str) -> int:
        return self._var_index[name]

    # ---------------- Build Mathematics Model ----------------
    @timeit_if_debug
    def build_model(self):
        """
        整体建模入口：创建 MOSEK Task 模型，调用子函数建变量、添加约束、目标函数
        """
        self.model = mosek.Task()

        super().build_model()

    @timeit_if_debug
    def create_variables(self):
        """
        创建决策变量（仅变量创建，不添加约束）：
          - 第一阶段: X[φ], Y[φ] (>=0)
          - 第二阶段对偶: r, s_i, t_i (<=0), l (<=0)
          - 线性决策规则系数: G0[φ,t,p] (prob alloc coeff), Gz[φ,t,p,i], Gu[φ,t,p,k]
          - R 系数: R0[φ,t], Rz[φ,t,i], Ru[φ,t,k]
          - π_q: 对于每个 q, 创建 2*I1+2 个对偶分量 self.pi[q][0..2I1+1]
        """
        # Stage I
        self.X_idx = {phi: self._add_var(f"X_{phi}") for phi in self.phi_list}
        self.Y_idx = {phi: self._add_var(f"Y_{phi}") for phi in self.phi_list}

        # Stage II duals
        self.r_idx = self._add_var("r")
        self.s_idx = {i: self._add_var(f"s_{i}") for i in range(self.I1)}
        self.t_idx = {k: self._add_var(f"t_{k}") for k in range(self.I1)}
        self.l_idx = self._add_var("l")

        # LDR G
        self.G0_idx = {}
        self.Gz_idx = {}
        self.Gu_idx = {}
        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    self.G0_idx[(phi, t, p)] = self._add_var(f"G0_{phi}_{t}_{p}")
                    for i in range(self.I1):
                        self.Gz_idx[(phi, t, p, i)] = self._add_var(f"Gz_{phi}_{t}_{p}_{i}")
                    for k in range(self.I1):
                        self.Gu_idx[(phi, t, p, k)] = self._add_var(f"Gu_{phi}_{t}_{p}_{k}")

        # LDR R
        self.R0_idx = {}
        self.Rz_idx = {}
        self.Ru_idx = {}
        for phi in self.phi_list:
            for t in self.t_list:
                self.R0_idx[(phi, t)] = self._add_var(f"R0_{phi}_{t}")
                for i in range(self.I1):
                    self.Rz_idx[(phi, t, i)] = self._add_var(f"Rz_{phi}_{t}_{i}")
                for k in range(self.I1):
                    self.Ru_idx[(phi, t, k)] = self._add_var(f"Ru_{phi}_{t}_{k}")

        # π_q
        self.pi_idx = {}
        for q in self.Q_list:
            self.pi_idx[q] = [self._add_var(f"pi_{q}_{j}") for j in range(3*self.I1+3)]

        self.num_vars = self._next_var
        self.model.appendvars(self.num_vars)

        # 变量上下界
        self.set_variable_bounds()

    def set_variable_bounds(self):
    # 设置所有变量为自由变量（lb=-inf, ub=+inf）
        for i in range(self.num_vars):
            self.model.putvarbound(i, mosek.boundkey.fr, -0.0, +0.0)
        # (26): t_k <= 0, l <= 0
        for k in range(self.I1):
            self.model.putvarbound(self.t_idx[k], mosek.boundkey.up, -0.0, 0.0)
        self.model.putvarbound(self.l_idx, mosek.boundkey.up, -0.0, 0.0)
        # (27): X[phi] >= 0, Y[phi] >= 0
        for phi in self.phi_list:
            self.model.putvarbound(self.X_idx[phi], mosek.boundkey.lo, 0.0, +0.0)
            self.model.putvarbound(self.Y_idx[phi], mosek.boundkey.lo, 0.0, +0.0)

    @timeit_if_debug
    def set_objective(self):
        """
          max:
            - Stage I:  Σ_φ p^hat_φ X_φ
            - Stage II:  r + sum_i s_i * mu_i + sum_i t_i * sigma_sq_i + l * (1^T Σ 1)
        """
        # c_j for all variables
        c = [0.0] * self._next_var

        # Stage I: p_hat[phi] * X[phi]
        for phi in self.phi_list:
            c[self.X_idx[phi]] = self.p_hat[phi]

        # Stage II: r + sum s_i mu_i + sum t_k sigma_sq_k + l * cost_cov
        c[self.r_idx] = 1.0
        for i in range(self.I1):
            c[self.s_idx[i]] = self.mu[i]
        for k in range(self.I1):
            c[self.t_idx[k]] = self.sigma_sq[k]
        c[self.l_idx] = self.cost_cov

        # self.model.putclist(range(self._next_var), c)
        self.model.putobjsense(mosek.objsense.maximize)

    @timeit_if_debug
    def add_constraints(self):
        """
        添加全体约束的主入口：
          - 第一阶段约束 (capacity)
          - Δ 与 R 之间的关系（表达式）及“ Δ/R=0 ”
          - 对每一个 q: 构造 α/γ 表达式并建立 SOCP 块
                （C^T π = α_z, D^T π = α_u, d^T π = γ, h^T π ≤ -α0, E^T π = 0）
          - 对每个 π_q 添加锥约束 (||x_1:n-1||2 ≤ x_n)
        """
        # # 1) 第一阶段约束
        self.add_first_stage_constraints()

        # 2) LDR: delta & R 关系（添加 Δ = 0/ R = 0 约束）
        self.set_delta_and_R()
        # # 3) 对每个 q: 构造 alpha/gamma，并添加 SOCP 对偶约束块
        for q in self.Q_list:
            self.add_SOCP_block(q)

    def add_first_stage_constraints(self):
        """
        第一阶段网络容量约束：
          对于每条边 e=(n,n') in A_prime: Σ_{φ: e∈paths[φ]} (X_φ + Y_φ) ≤ capacity_e
        添加位置：模型中第一阶段约束（与论文中的可行集 X 一致）
        """
        for edge, capacity in self.A_prime.items():
            vars_ = []
            coeffs = []
            for phi in self.phi_list:
                if edge in self.paths.get(phi, []):
                    vars_.append(self.X_idx[phi])
                    coeffs.append(1.0)
                    vars_.append(self.Y_idx[phi])
                    coeffs.append(1.0)
            if vars_:
                self.model.appendcons(1)
                con_idx = self.model.getnumcon() - 1
                self.model.putarow(con_idx, vars_, coeffs)
                self.model.putconbound(con_idx, mosek.boundkey.up, -0.0, capacity)

    def set_delta_and_R(self):
        """
        构造 Δ 表达式，并依据 t_d_phi 添加零约束：
        论文形式：
          Δ^0_{φt}          = R^0_{φt} - Y_φ + Σ_{t'<t} ( d^0_{φt'} - a Σ_p p G^0_{φt'p} - a p̂_φ )
          Δ^{(z)}_{φt,i}  = R^{(z)}_{φt,i} + Σ_{t'<t} ( d^{(z)}_{φt',i} - a Σ_p p G^{(z)}_{φt'p,i} )
          Δ^{(u)}_{φt,k} = R^{(u)}_{φt,k} - a Σ_{t'<t} Σ_p p G^{(u)}_{φt'p,k}
        约束：
          1 ≤t ≤ t_d_phi[φ] :  Δ^*_{φt,*} == 0 （需求期内，递推公式带入LDR，这些量为0）
          t > t_d                  :  R^*_{φt,*} == 0 （超出需求期，R 置 0）
        实现：
          - Δ 以 LinExpr 存储 self.delta0[(φ,t)] 等
          - 直接对 Δ 或 R 添加等式约束
        """
        for phi in self.phi_list:
            t_deadline = self.t_d_phi.get(phi, 0)  # 获取该路径的需求截止时间
            for t in self.t_list:
                if 1 <= t <= t_deadline:
                    print(f"  -> Adding Δ=0 constraints for (φ ={phi}, t={t})")
                    # (14): R0 - sum_{t'<t} a sum_p p G0 - Y = - sum_{t'<t} (d0 + a p_hat)
                    self.add_delta0_eq_0(phi, t)

                    # (15): Rz + sum (d_z - a sum p Gz) = 0
                    self.add_delta_z_eq_0(phi, t)

                    # (16): Ru - a sum p Gu = 0
                    self.add_delta_u_eq_0(phi, t)

                elif t > t_deadline:
                    # (17)–(19): R = 0
                    print(f"  -> Adding R=0 constraints for (φ ={phi}, t={t})")
                    self.add_R_0_eq_0(phi, t)
                    # R^z = 0
                    self.add_R_z_eq_0(phi, t)
                    # R^u = 0
                    self.add_R_u_eq_0(phi, t)

    def add_delta0_eq_0(self, phi, t):
        """约束 (14)"""
        # (14): R0 - sum_{t'<t} a sum_p p G0 - Y = - sum_{t'<t} (d0 + a p_hat)
        vars_ = [self.R0_idx[(phi, t)], self.Y_idx[phi]]
        coeffs = [1.0, -1.0]
        constant = 0.0       # - sum_{t'<t} (d0 + a p_hat)
        for tp in self.t_list:
                        if tp < t:
                            d0_val = self.d_0_phi_t.get((phi, tp), 0.0)
                            constant += (d0_val + self.a * self.p_hat.get(phi, 0.0))
                            for p in self.p_list:
                                vars_.append(self.G0_idx[(phi, tp, p)])
                                coeffs.append(-self.a * p)
                        else:
                            break
        self.model.appendcons(1)
        con_idx = self.model.getnumcon() - 1
        self.model.putarow(con_idx, vars_, coeffs)
        self.model.putconbound(con_idx, mosek.boundkey.fx, -constant, -constant)

    def add_delta_z_eq_0(self, phi, t):
                    for i in range(self.I1):
                        vars_ = [self.Rz_idx[(phi, t, i)]]
                        coeffs = [1.0]
                        constant = 0.0
                        for tp in self.t_list:
                            if tp < t:
                                dz_val = self.d_z_phi_t_i.get((phi, tp, i), 0.0)
                                constant += dz_val
                                for p in self.p_list:
                                    vars_.append(self.Gz_idx[(phi, tp, p, i)])
                                    coeffs.append(-self.a * p)
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, vars_, coeffs)
                        self.model.putconbound(con_idx, mosek.boundkey.fx, -constant, -constant)

    def add_delta_u_eq_0(self, phi, t):
                    for k in range(self.I1):
                        vars_ = [self.Ru_idx[(phi, t, k)]]
                        coeffs = [1.0]
                        rhs = 0.0
                        for tp in self.t_list:
                            if tp < t:
                                for p in self.p_list:
                                    vars_.append(self.Gu_idx[(phi, tp, p, k)])
                                    coeffs.append(-self.a * p)
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, vars_, coeffs)
                        self.model.putconbound(con_idx, mosek.boundkey.fx, rhs, rhs)

    def add_R_0_eq_0(self, phi, t):
                    self.model.appendcons(1)
                    con_idx = self.model.getnumcon() - 1
                    self.model.putarow(con_idx, [self.R0_idx[(phi, t)]], [1.0])
                    self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

    def add_R_z_eq_0(self, phi, t):
                    for i in range(self.I1):
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, [self.Rz_idx[(phi, t, i)]], [1.0])
                        self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

    def add_R_u_eq_0(self, phi, t):
                    for k in range(self.I1):
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, [self.Ru_idx[(phi, t, k)]], [1.0])
                        self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

    def add_SOCP_block(self, q):
        """约束 (20)–(25) for each q"""
        # 1) C^T π_q = α^{q,(z)}   (I1 条等式)
        # 2) D^T π_q = α^{q,(u)}   (I1 条等式)
        # 3) d^T π_q = γ_q              (1 条等式)
        # 4) E^T π_q = 0                  (I1 + 1 条等式)
        # 5) h^T π_q ≤ - α^q_0        (1 条不等式)
        self.set_dual_linear_constr(q)

        # 6) π_q ⪰_K 0 : 对每对 (3i, 3i+1) 使用 Norm 约束，最后一对 (3I1, 3I1+1) 也一样
        self.set_second_order_constr(q)


    def set_second_order_constr(self, q):
        # (25): SOC constraints — I1+1 个 3D 锥
        for i in range(self.I1):
            # || [pi[3i+1], pi[3i+2]] ||_2 <= pi[3i]
            self.add_soc_constraint(self.pi_idx[q][3*i], [self.pi_idx[q][3*i+1], self.pi_idx[q][3*i+2]])

        # 聚合锥
        self.add_soc_constraint(self.pi_idx[q][3*self.I1], [self.pi_idx[q][3*self.I1+1], self.pi_idx[q][3*self.I1+2]])

    def add_soc_constraint(self, t_var_idx, y_var_indices):
        """
        添加二阶锥约束: ||y||_2 <= t

        参数:
            t_var_idx (int): 标量变量 t 的 MOSEK 变量索引（必须是单个整数）
            y_var_indices (list of int): 向量 y 的 MOSEK 变量索引列表，如 [y0, y1, ..., y_{n-1}]

        效果:
            向 self.model 添加一个标准二次锥约束：
                [t, y0, y1, ..., y_{n-1}] ∈ Q^{n+1}
        """
        n = len(y_var_indices)
        if n == 0:
            raise ValueError("y_var_indices cannot be empty")

        # 获取当前 AFE 数量，并追加 (n+1) 个 AFE
        afe_base = self.model.getnumafe()
        self.model.appendafes(n + 1)

        # AFE[0] = t
        self.model.putafefentry(afe_base, t_var_idx, 1.0)
        self.model.putafeg(afe_base, 0.0)

        # AFE[1..n] = y[0..n-1]
        for i, y_idx in enumerate(y_var_indices):
            self.model.putafefentry(afe_base + 1 + i, y_idx, 1.0)
            self.model.putafeg(afe_base + 1 + i, 0.0)

        quad_dom = self.model.appendquadraticconedomain(n + 1)
        afe_list = list(range(afe_base, afe_base + n + 1))
        self.model.appendacc(quad_dom, afe_list, None)


        # 辅助函数  将表达式 e 合并到 base_vars/base_coeffs 中，带符号
    def _merge_expr(self, base_vars, base_coeffs, expr, sign=1.0):
            """返回 (all_vars, all_coeffs, total_const) for: base + sign * e"""
            vars_ = list(base_vars)
            coeffs_ = list(base_coeffs)
            const = expr['const'] * sign
            for var_idx, coeff in expr['coeffs'].items():
                vars_.append(var_idx)
                coeffs_.append(coeff * sign)
            return vars_, coeffs_, const

    def _add_eq_constraint(self, vars_, coeffs, rhs):
        self.model.appendcons(1)
        con_idx = self.model.getnumcon() - 1
        self.model.putarow(con_idx, vars_, coeffs)
        self.model.putconbound(con_idx, mosek.boundkey.fx, rhs, rhs)

    def _add_le_constraint(self, vars_, coeffs, rhs):
        self.model.appendcons(1)
        con_idx = self.model.getnumcon() - 1
        self.model.putarow(con_idx, vars_, coeffs)
        self.model.putconbound(con_idx, mosek.boundkey.up, -self.INF, rhs)


    def set_alpha_and_gamma(self, q):
        """
        返回 alpha0, alpha_z, alpha_u, gamma 的表达式描述
        每个表达式是 {'const': c, 'coeffs': {var_idx: coeff}}
        """
        I1 = self.I1

        def expr(const=0.0, coeffs=None):
            return {'const': float(const), 'coeffs': coeffs or {}}

        if q == 'obj':
            # alpha0 = r - sum c G0
            coeffs0 = {self.r_idx: 1.0}
            for phi in self.phi_list:
                for t in self.t_list:
                    for p in self.p_list:
                        c_val = self.c_phi_tp.get((phi, t, p), 0.0)
                        if c_val != 0:
                            var = self.G0_idx[(phi, t, p)]
                            coeffs0[var] = coeffs0.get(var, 0.0) - c_val
            alpha0 = expr(const=0.0, coeffs=coeffs0)

            # alpha_z[i] = s[i] - sum c Gz
            alpha_z = []
            for i in range(I1):
                coeffs_z = {self.s_idx[i]: 1.0}
                for phi in self.phi_list:
                    for t in self.t_list:
                        for p in self.p_list:
                            c_val = self.c_phi_tp.get((phi, t, p), 0.0)
                            if c_val != 0:
                                var = self.Gz_idx[(phi, t, p, i)]
                                coeffs_z[var] = coeffs_z.get(var, 0.0) - c_val
                alpha_z.append(expr(const=0.0, coeffs=coeffs_z))

            # alpha_u[k] = t[k] - sum c Gu
            alpha_u = []
            for k in range(I1):
                coeffs_u = {self.t_idx[k]: 1.0}
                for phi in self.phi_list:
                    for t in self.t_list:
                        for p in self.p_list:
                            c_val = self.c_phi_tp.get((phi, t, p), 0.0)
                            if c_val != 0:
                                var = self.Gu_idx[(phi, t, p, k)]
                                coeffs_u[var] = coeffs_u.get(var, 0.0) - c_val
                alpha_u.append(expr(const=0.0, coeffs=coeffs_u))

            # gamma = l
            gamma = expr(const=0.0, coeffs={self.l_idx: 1.0})

        else:
            typ = q[0]
            if typ == 'svc':
                phi, t = q[1], q[2]
                # alpha0 = d0 - a Σ p G0 - R0 + a p_hat
                const0 = self.d_0_phi_t.get((phi, t), 0.0) + self.a * self.p_hat.get(phi, 0.0)
                coeffs0 = {self.R0_idx[(phi, t)]: -1.0}
                for p in self.p_list:
                    var = self.G0_idx[(phi, t, p)]
                    coeffs0[var] = coeffs0.get(var, 0.0) - self.a * p
                alpha0 = expr(const=const0, coeffs=coeffs0)

                # alpha_z[i] = dz - a Σ p Gz - Rz
                alpha_z = []
                for i in range(I1):
                    const_z = self.d_z_phi_t_i.get((phi, t, i), 0.0)
                    coeffs_z = {self.Rz_idx[(phi, t, i)]: -1.0}
                    for p in self.p_list:
                        var = self.Gz_idx[(phi, t, p, i)]
                        coeffs_z[var] = coeffs_z.get(var, 0.0) - self.a * p
                    alpha_z.append(expr(const=const_z, coeffs=coeffs_z))

                # alpha_u[k] = -a Σ p Gu - Ru
                alpha_u = []
                for k in range(I1):
                    coeffs_u = {self.Ru_idx[(phi, t, k)]: -1.0}
                    for p in self.p_list:
                        var = self.Gu_idx[(phi, t, p, k)]
                        coeffs_u[var] = coeffs_u.get(var, 0.0) - self.a * p
                    alpha_u.append(expr(const=0.0, coeffs=coeffs_u))

                gamma = expr(const=0.0, coeffs={})

            elif typ == 'mix':
                phi, t = q[1], q[2]
                # alpha0 = Σ G0 - 1
                coeffs0 = {}
                for p in self.p_list:
                    var = self.G0_idx[(phi, t, p)]
                    coeffs0[var] = coeffs0.get(var, 0.0) + 1.0
                alpha0 = expr(const=-1.0, coeffs=coeffs0)

                # alpha_z[i] = Σ Gz
                alpha_z = []
                for i in range(I1):
                    coeffs_z = {}
                    for p in self.p_list:
                        var = self.Gz_idx[(phi, t, p, i)]
                        coeffs_z[var] = coeffs_z.get(var, 0.0) + 1.0
                    alpha_z.append(expr(const=0.0, coeffs=coeffs_z))

                # alpha_u[k] = Σ Gu
                alpha_u = []
                for k in range(I1):
                    coeffs_u = {}
                    for p in self.p_list:
                        var = self.Gu_idx[(phi, t, p, k)]
                        coeffs_u[var] = coeffs_u.get(var, 0.0) + 1.0
                    alpha_u.append(expr(const=0.0, coeffs=coeffs_u))

                gamma = expr(const=0.0, coeffs={})

            elif typ == 'ng':
                phi, t, p = q[1], q[2], q[3]
                alpha0 = expr(const=0.0, coeffs={self.G0_idx[(phi, t, p)]: -1.0})
                alpha_z = [expr(const=0.0, coeffs={self.Gz_idx[(phi, t, p, i)]: -1.0}) for i in range(I1)]
                alpha_u = [expr(const=0.0, coeffs={self.Gu_idx[(phi, t, p, k)]: -1.0}) for k in range(I1)]
                gamma = expr(const=0.0, coeffs={})

            elif typ == 'nr':
                phi, t = q[1], q[2]
                alpha0 = expr(const=0.0, coeffs={self.R0_idx[(phi, t)]: -1.0})
                alpha_z = [expr(const=0.0, coeffs={self.Rz_idx[(phi, t, i)]: -1.0}) for i in range(I1)]
                alpha_u = [expr(const=0.0, coeffs={self.Ru_idx[(phi, t, k)]: -1.0}) for k in range(I1)]
                gamma = expr(const=0.0, coeffs={})

            else:
                raise ValueError(f"Unknown q: {q}")

        return alpha0, alpha_z, alpha_u, gamma

    def set_dual_linear_constr(self, q):
        alpha0, alpha_z, alpha_u, gamma = self.set_alpha_and_gamma(q)
        pi_vars = self.pi_idx[q]

        # (20) C^T pi = alpha_z[i]  → C^T pi - alpha_z[i] = 0
        for i in range(self.I1):
            base_vars = pi_vars
            base_coeffs = self.C[:, i].tolist()
            vars_, coeffs_, const = self._merge_expr(base_vars, base_coeffs, alpha_z[i], sign=-1.0)
            self._add_eq_constraint(vars_, coeffs_, -const)

        # (21) D^T pi = alpha_u[k] → D^T pi - alpha_u[k] = 0
        for k in range(self.I1):
            base_vars = pi_vars
            base_coeffs = self.D[:, k].tolist()
            vars_, coeffs_, const = self._merge_expr(base_vars, base_coeffs, alpha_u[k], sign=-1.0)
            self._add_eq_constraint(vars_, coeffs_, -const)

        # (22) d^T pi = gamma → d^T pi - gamma = 0
        base_vars = pi_vars
        base_coeffs = self.d.tolist()
        vars_, coeffs_, const = self._merge_expr(base_vars, base_coeffs, gamma, sign=-1.0)
        self._add_eq_constraint(vars_, coeffs_, -const)

        # (24) h^T pi + alpha0 <= 0 → h^T pi + alpha0 <= 0
        base_vars = pi_vars
        base_coeffs = self.h.tolist()
        vars_, coeffs_, const = self._merge_expr(base_vars, base_coeffs, alpha0, sign=1.0)
        self._add_le_constraint(vars_, coeffs_, -const)


    def solve(self, time_limit: float = 300.0, rel_gap: float = 1e-3, verbose: bool = True):
        # 设置求解参数
        self.model.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, rel_gap)
        self.model.putdouparam(mosek.dparam.optimizer_max_time, time_limit)

        if verbose:
            # 启用日志输出到终端
            def streamprinter(text):
                sys.stdout.write(text)
                sys.stdout.flush()

            self.model.set_Stream(mosek.streamtype.log, streamprinter)

        # 开始求解
        self.model.optimize()

        if verbose:
            self.model.solutionsummary(mosek.streamtype.log)

        self.print_model_status()

    def get_solution(self):
        """
        从已成功求解的 SOCP4LDR_Mosek 实例中提取所有变量的解。

        该函数严格遵循 Manuscript_XX_20250904 -LJ.pdf 附录 D 中的变量定义。

        参数:
        - socp_instance: 已调用 solve() 并成功求解的 SOCP4LDR_Mosek 类实例。

        返回:
        - solutions: 一个包含所有解的嵌套字典。
        """
        # 检查求解状态
        status = self.get_status()
        if status != "OPTIMAL":
            raise ValueError(f"Model has not been solved to optimality. Current status: {status}")

        # 获取原始解向量
        xx = [0.0] * self._next_var
        self.model.getxx(mosek.soltype.itr, xx)

        solutions = {}

        # --- 1. 提取第一阶段变量 ---
        solutions['X'] = {phi: xx[self.X_idx[phi]] for phi in self.phi_list}
        solutions['Y'] = {phi: xx[self.Y_idx[phi]] for phi in self.phi_list}

        # --- 2. 提取第二阶段对偶变量 (r, s, t, l) ---
        solutions['r'] = xx[self.r_idx]
        solutions['s'] = [xx[self.s_idx[i]] for i in range(self.I1)]
        solutions['t'] = [xx[self.t_idx[k]] for k in range(self.I1)]
        solutions['l'] = xx[self.l_idx]

        # --- 3. 提取 LDR 系数 (G 和 R) ---
        solutions['G0'] = {}
        solutions['Gz'] = {}
        solutions['Gu'] = {}
        solutions['R0'] = {}
        solutions['Rz'] = {}
        solutions['Ru'] = {}

        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    solutions['G0'][(phi, t, p)] = xx[self.G0_idx[(phi, t, p)]]
                    for i in range(self.I1):
                        solutions['Gz'][(phi, t, p, i)] = xx[self.Gz_idx[(phi, t, p, i)]]
                    for k in range(self.I1):
                        solutions['Gu'][(phi, t, p, k)] = xx[self.Gu_idx[(phi, t, p, k)]]

                solutions['R0'][(phi, t)] = xx[self.R0_idx[(phi, t)]]
                for i in range(self.I1):
                    solutions['Rz'][(phi, t, i)] = xx[self.Rz_idx[(phi, t, i)]]
                for k in range(self.I1):
                    solutions['Ru'][(phi, t, k)] = xx[self.Ru_idx[(phi, t, k)]]

        # --- 4. 提取对偶变量 pi_q ---
        solutions['pi'] = {}
        for q in self.Q_list:
            solutions['pi'][q] = [xx[idx] for idx in self.pi_idx[q]]

        # --- 5. 提取目标函数值 ---
        solutions['obj_val'] = self.model.getprimalobj(mosek.soltype.itr)

        # --- 6. (可选) 提取 alpha 系数的值 ---
        # 利用类内已实现的 get_alpha_values 方法
        solutions['alpha'] = self.get_alpha_values()

        return solutions

    def get_pi_solution(self):
        """
        返回对偶变量 π_q 的解（注意：在本模型中 π_q 是原始变量，非 MOSEK 对偶变量）。
        返回格式: { q: [pi_0, pi_1, ..., pi_{3*I1+2}] for q in Q_list }
        """
        status = self.get_status()
        if status != "OPTIMAL":
            logging.warning("Model not solved to optimality. Returning None for pi.")
            return None

        # 获取原始解向量
        xx = [0.0] * self._next_var
        self.model.getxx(mosek.soltype.itr, xx)

        pi_solution = {}
        for q in self.Q_list:
            pi_vals = [xx[idx] for idx in self.pi_idx[q]]
            pi_solution[q] = pi_vals

        return pi_solution

    def get_alpha_values(self):
        """
        根据 MOSEK 模型的最优解，计算每个 q ∈ Q 对应的 alpha 系数 (alpha0, alpha_z, alpha_u, gamma)。

        返回:
            alpha_values: dict, key=q, value=dict(alpha0, alpha_z, alpha_u, gamma)
        """
        status = self.get_status()
        if status != "OPTIMAL":
            logging.warning("Model not solved to optimality. Cannot extract alpha values.")
            return None

        # 1. 获取所有变量的解
        xx = [0.0] * self._next_var
        self.model.getxx(mosek.soltype.itr, xx)

        # 辅助函数：从解向量中获取变量值
        def get_var_val(name_or_idx):
            if isinstance(name_or_idx, str):
                idx = self._var_index[name_or_idx]
            else:
                idx = name_or_idx
            return xx[idx]

        # 2. 提取 Stage II duals
        r_val = get_var_val(self.r_idx)
        s_vals = [get_var_val(self.s_idx[i]) for i in range(self.I1)]
        t_vals = [get_var_val(self.t_idx[k]) for k in range(self.I1)]
        l_val = get_var_val(self.l_idx)

        # 3. 提取 LDR 系数
        G0_vals = {}
        Gz_vals = {}
        Gu_vals = {}
        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    G0_vals[(phi, t, p)] = get_var_val(self.G0_idx[(phi, t, p)])
                    for i in range(self.I1):
                        Gz_vals[(phi, t, p, i)] = get_var_val(self.Gz_idx[(phi, t, p, i)])
                    for k in range(self.I1):
                        Gu_vals[(phi, t, p, k)] = get_var_val(self.Gu_idx[(phi, t, p, k)])

        R0_vals = {}
        Rz_vals = {}
        Ru_vals = {}
        for phi in self.phi_list:
            for t in self.t_list:
                R0_vals[(phi, t)] = get_var_val(self.R0_idx[(phi, t)])
                for i in range(self.I1):
                    Rz_vals[(phi, t, i)] = get_var_val(self.Rz_idx[(phi, t, i)])
                for k in range(self.I1):
                    Ru_vals[(phi, t, k)] = get_var_val(self.Ru_idx[(phi, t, k)])

        # 4. 初始化结果字典
        alpha_values = {}

        # --- (1) 'obj' 类型 ---
        q = 'obj'
        alpha0_val = r_val - sum(
            self.c_phi_tp.get((phi, t, p), 0.0) * G0_vals[(phi, t, p)]
            for phi in self.phi_list for t in self.t_list for p in self.p_list
        )
        alpha_z_vals = [
            s_vals[i] - sum(
                self.c_phi_tp.get((phi, t, p), 0.0) * Gz_vals[(phi, t, p, i)]
                for phi in self.phi_list for t in self.t_list for p in self.p_list
            )
            for i in range(self.I1)
        ]
        alpha_u_vals = [
            t_vals[k] - sum(
                self.c_phi_tp.get((phi, t, p), 0.0) * Gu_vals[(phi, t, p, k)]
                for phi in self.phi_list for t in self.t_list for p in self.p_list
            )
            for k in range(self.I1)
        ]
        gamma_val = l_val

        alpha_values[q] = {
            'alpha0': alpha0_val,
            'alpha_z': alpha_z_vals,
            'alpha_u': alpha_u_vals,
            'gamma': gamma_val
        }

        # --- (2) 其他 q 类型 ---
        for phi in self.phi_list:
            for t in self.t_list:
                # (a) 'svc'
                q = ('svc', phi, t)
                alpha0_val = (
                    self.d_0_phi_t.get((phi, t), 0.0)
                    - self.a * sum(p * G0_vals[(phi, t, p)] for p in self.p_list)
                    - R0_vals[(phi, t)]
                    + self.a * self.p_hat.get(phi, 0.0)
                )
                alpha_z_vals = [
                    self.d_z_phi_t_i.get((phi, t, i), 0.0)
                    - self.a * sum(p * Gz_vals[(phi, t, p, i)] for p in self.p_list)
                    - Rz_vals[(phi, t, i)]
                    for i in range(self.I1)
                ]
                alpha_u_vals = [
                    -self.a * sum(p * Gu_vals[(phi, t, p, k)] for p in self.p_list)
                    - Ru_vals[(phi, t, k)]
                    for k in range(self.I1)
                ]
                alpha_values[q] = {
                    'alpha0': alpha0_val,
                    'alpha_z': alpha_z_vals,
                    'alpha_u': alpha_u_vals,
                    'gamma': 0.0
                }

                # (b) 'mix'
                q = ('mix', phi, t)
                alpha0_val = sum(G0_vals[(phi, t, p)] for p in self.p_list) - 1.0
                alpha_z_vals = [
                    sum(Gz_vals[(phi, t, p, i)] for p in self.p_list)
                    for i in range(self.I1)
                ]
                alpha_u_vals = [
                    sum(Gu_vals[(phi, t, p, k)] for p in self.p_list)
                    for k in range(self.I1)
                ]
                alpha_values[q] = {
                    'alpha0': alpha0_val,
                    'alpha_z': alpha_z_vals,
                    'alpha_u': alpha_u_vals,
                    'gamma': 0.0
                }

                # (c) 'ng'
                for p in self.p_list:
                    q = ('ng', phi, t, p)
                    alpha_values[q] = {
                        'alpha0': -G0_vals[(phi, t, p)],
                        'alpha_z': [-Gz_vals[(phi, t, p, i)] for i in range(self.I1)],
                        'alpha_u': [-Gu_vals[(phi, t, p, k)] for k in range(self.I1)],
                        'gamma': 0.0
                    }

                # (d) 'nr'
                q = ('nr', phi, t)
                alpha_values[q] = {
                    'alpha0': -R0_vals[(phi, t)],
                    'alpha_z': [-Rz_vals[(phi, t, i)] for i in range(self.I1)],
                    'alpha_u': [-Ru_vals[(phi, t, k)] for k in range(self.I1)],
                    'gamma': 0.0
                }

        return alpha_values

    def get_status(self):
        """
        获取 MOSEK 模型求解状态
        """
        try:
            # 获取问题状态和解状态（使用 interior-point solution）
            prosta = self.model.getprosta(mosek.soltype.itr)
            solsta = self.model.getsolsta(mosek.soltype.itr)
        except mosek.Error:
            return "ERROR"

        self.model.writedata(self.info + '.ptf')

        if solsta == mosek.solsta.optimal:
            return "OPTIMAL"
        elif prosta == mosek.prosta.prim_infeas:
            return "INFEASIBLE"
        elif prosta == mosek.prosta.dual_infeas:
            return "UNBOUNDED"
        elif solsta in [mosek.solsta.near_optimal, mosek.solsta.unknown]:
            # 可能是时间限制、数值问题等
            # 检查是否达到时间限制（需用户自行记录）
            return "TIME_LIMIT_OR_NEAR_OPTIMAL"
        else:
            return f"OTHER (prosta={prosta}, solsta={solsta})"

    def print_model_status(self):
        """
        打印 MOSEK 模型状态，并在不可行时导出 .ptf 文件（MOSEK 不支持 .ilp）
        """
        try:
            prosta = self.model.getprosta(mosek.soltype.itr)
            solsta = self.model.getsolsta(mosek.soltype.itr)
        except mosek.Error as e:
            logging.error(f"Failed to get solution status: {e}")
            return

        if solsta == mosek.solsta.optimal:
            return  # 正常，无需处理

        file_name = self.info

        if prosta == mosek.prosta.prim_infeas:
            # 原始不可行：导出问题文件用于调试
            self.model.writedata(file_name + '_infeasible.ptf')
            self.model.writedata(file_name + '.ptf')
            logging.warning("模型无可行解 (Primal Infeasible)")

        elif prosta == mosek.prosta.dual_infeas:
            # 对偶不可行 → 通常意味着原始问题无界
            self.model.writedata(file_name + '_unbounded.ptf')
            logging.warning("模型无界 (Dual Infeasible)")

        elif solsta in [mosek.solsta.near_optimal, mosek.solsta.unknown]:
            # 可能是时间限制或数值问题
            try:
                obj_val = self.model.getprimalobj(mosek.soltype.itr)
                logging.warning(f"求解未完成（可能超时），当前目标值：{obj_val}")
            except:
                logging.warning("求解未完成，无有效目标值")

        else:
            logging.warning(f"其他状态：prosta={prosta}, solsta={solsta}")

    def print_model_info(self):
        """
        打印 模型的基本信息：变量数量和约束数量。
        """
        pass


    def write(self):
        self.model.writedata(self.info + '.ptf')
