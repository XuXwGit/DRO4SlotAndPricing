"""
使用Mosek 建模 重构后的SOCP模型，由Qwen转换 LDR_based_SOCP.py 得到
"""
import numpy as np
import sys
from typing import Dict, Any, List
import os
import sys
import mosek

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_builder import ModelBuilder
from src.utils.model_params import generate_feasible_test_case

class SOCP4LDR_Mosek(ModelBuilder):
    """
    使用 MOSEK Optimizer API (Task) 实现完整 DRP 问题：
        max_{X,Y} sum_phi p_hat[phi] * X[phi] + beta_E^LDR(X, Y)

    其中 beta_E^LDR 由 SOCP (A65) 给出。
    """

    def __init__(self, model_params: Dict[str, Any], debug: bool = False):
        self.debug = debug
        super().__init__(info="SOCP4LDR", solver="mosek")
        self.set_model_params(model_params)

        # 变量索引映射
        self._var_index = {}
        self._next_var = 0

    def _add_var(self, name: str) -> int:
        idx = self._next_var
        self._var_index[name] = idx
        self._next_var += 1
        return idx

    def _get_var(self, name: str) -> int:
        return self._var_index[name]

    def build_model(self):
        """构建 MOSEK Task 模型"""
        self.model = mosek.Task()

        self.create_variables()
        self.set_objective()
        self.add_constraints()

    def create_variables(self):
        I1 = self.I1
        # Stage I
        self.X_idx = {phi: self._add_var(f"X_{phi}") for phi in self.phi_list}
        self.Y_idx = {phi: self._add_var(f"Y_{phi}") for phi in self.phi_list}

        # Stage II duals
        self.r_idx = self._add_var("r")
        self.s_idx = {i: self._add_var(f"s_{i}") for i in range(I1)}
        self.t_idx = {k: self._add_var(f"t_{k}") for k in range(I1)}
        self.l_idx = self._add_var("l")

        # LDR G
        self.G0_idx = {}
        self.Gz_idx = {}
        self.Gu_idx = {}
        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    self.G0_idx[(phi, t, p)] = self._add_var(f"G0_{phi}_{t}_{p}")
                    for i in range(I1):
                        self.Gz_idx[(phi, t, p, i)] = self._add_var(f"Gz_{phi}_{t}_{p}_{i}")
                    for k in range(I1):
                        self.Gu_idx[(phi, t, p, k)] = self._add_var(f"Gu_{phi}_{t}_{p}_{k}")

        # LDR R
        self.R0_idx = {}
        self.Rz_idx = {}
        self.Ru_idx = {}
        for phi in self.phi_list:
            for t in self.t_list:
                self.R0_idx[(phi, t)] = self._add_var(f"R0_{phi}_{t}")
                for i in range(I1):
                    self.Rz_idx[(phi, t, i)] = self._add_var(f"Rz_{phi}_{t}_{i}")
                for k in range(I1):
                    self.Ru_idx[(phi, t, k)] = self._add_var(f"Ru_{phi}_{t}_{k}")

        # π_q
        self.pi_idx = {}
        for q in self.Q_list:
            self.pi_idx[q] = [self._add_var(f"pi_{q}_{j}") for j in range(3*I1+3)]

        num_vars = self._next_var
        self.model.appendvars(num_vars)

        # 变量上下界
        self.set_variable_bounds()

    def set_variable_bounds(self):
        # (26): t_k <= 0, l <= 0
        for k in range(self.I1):
            self.model.putvarbound(self.t_idx[k], mosek.boundkey.up, -0.0, 0.0)
        self.model.putvarbound(self.l_idx, mosek.boundkey.up, -0.0, 0.0)
        # (27): X[phi] >= 0, Y[phi] >= 0
        for phi in self.phi_list:
            self.model.putvarbound(self.X_idx[phi], mosek.boundkey.lo, 0.0, +0.0)
            self.model.putvarbound(self.Y_idx[phi], mosek.boundkey.lo, 0.0, +0.0)


    def set_objective(self):

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

        self.model.putclist(range(self._next_var), c)
        self.model.putobjsense(mosek.objsense.maximize)

    def add_constraints(self):
        # # 1) 第一阶段约束
        self.add_first_stage_constraints()

        # 2) delta & R 关系（构造表达式，添加 0/0 约束）
        self.set_delta_and_R()
        # 3) 对每个 q: 构造 alpha/gamma，并添加 SOCP 对偶约束块
        for q in self.Q_list:
            self.add_SOCP_block(q)

    def add_first_stage_constraints(self):
        """约束 (28): 网络容量"""
        for edge, cap in self.A_prime.items():
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
                self.model.putconbound(con_idx, mosek.boundkey.up, -0.0, cap)

    def set_delta_and_R(self):
        """约束 (14)–(19)"""
        for phi in self.phi_list:
            t_deadline = self.t_d_phi[phi]
            for t in self.t_list:
                if 1 <= t <= t_deadline:
                    # (14): R0 - sum_{t'<t} a sum_p p G0 = Y - sum_{t'<t} (d0 + a p_hat)
                    vars_ = [self.R0_idx[(phi, t)], self.Y_idx[phi]]
                    coeffs = [1.0, -1.0]
                    rhs = 0.0
                    for tp in self.t_list:
                        if tp < t:
                            d0_val = self.d_0_phi_t.get((phi, tp), 0.0)
                            rhs -= (d0_val + self.a * self.p_hat.get(phi, 0.0))
                            for p in self.p_list:
                                vars_.append(self.G0_idx[(phi, tp, p)])
                                coeffs.append(-self.a * p)
                    self.model.appendcons(1)
                    con_idx = self.model.getnumcon() - 1
                    self.model.putarow(con_idx, vars_, coeffs)
                    self.model.putconbound(con_idx, mosek.boundkey.fx, rhs, rhs)

                    # (15): Rz + sum (d_z - a sum p Gz) = 0
                    for i in range(self.I1):
                        vars_ = [self.Rz_idx[(phi, t, i)]]
                        coeffs = [1.0]
                        rhs = 0.0
                        for tp in self.t_list:
                            if tp < t:
                                dz_val = self.d_z_phi_t_i.get((phi, tp, i), 0.0)
                                rhs += dz_val
                                for p in self.p_list:
                                    vars_.append(self.Gz_idx[(phi, tp, p, i)])
                                    coeffs.append(-self.a * p)
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, vars_, coeffs)
                        self.model.putconbound(con_idx, mosek.boundkey.fx, rhs, rhs)

                    # (16): Ru - a sum p Gu = 0
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

                elif t > t_deadline:
                    # (17)–(19): R = 0
                    self.model.appendcons(1)
                    con_idx = self.model.getnumcon() - 1
                    self.model.putarow(con_idx, [self.R0_idx[(phi, t)]], [1.0])
                    self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

                    for i in range(self.I1):
                        self.model.appendcons(1)
                        con_idx = self.model.getnumcon() - 1
                        self.model.putarow(con_idx, [self.Rz_idx[(phi, t, i)]], [1.0])
                        self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

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
            # || [pi[3i], pi[3i+1]] ||_2 <= pi[3i+2]
            afe_idx = self.model.getnumafe()
            self.model.appendafes(3)
            # t - pi[3i+2] = 0
            self.model.putafefentry(afe_idx, self.pi_idx[q][3*i+2], -1.0)
            self.model.putafeg(afe_idx, 0.0)
            # x1 - pi[3i] = 0
            self.model.putafefentry(afe_idx+1, self.pi_idx[q][3*i], -1.0)
            self.model.putafeg(afe_idx+1, 0.0)
            # x2 - pi[3i+1] = 0
            self.model.putafefentry(afe_idx+2, self.pi_idx[q][3*i+1], -1.0)
            self.model.putafeg(afe_idx+2, 0.0)
            quad_dom = self.model.appendquadraticconedomain(3)
            self.model.appendacc(quad_dom, [afe_idx, afe_idx+1, afe_idx+2], None)

        # 聚合锥
        afe_idx = self.model.getnumafe()
        self.model.appendafes(3)
        self.model.putafefentry(afe_idx, self.pi_idx[q][3*self.I1+2], -1.0)
        self.model.putafeg(afe_idx, 0.0)
        self.model.putafefentry(afe_idx+1, self.pi_idx[q][3*self.I1], -1.0)
        self.model.putafeg(afe_idx+1, 0.0)
        self.model.putafefentry(afe_idx+2, self.pi_idx[q][3*self.I1+1], -1.0)
        self.model.putafeg(afe_idx+2, 0.0)
        quad_dom = self.model.appendquadraticconedomain(3)
        self.model.appendacc(quad_dom, [afe_idx, afe_idx+1, afe_idx+2], None)


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

    def get_solution(self):
        """提取解"""
        sol = {}
        xx = [0.0] * self._next_var
        self.model.getxx(mosek.soltype.itr, xx)

        sol['X'] = {phi: xx[self.X_idx[phi]] for phi in self.phi_list}
        sol['Y'] = {phi: xx[self.Y_idx[phi]] for phi in self.phi_list}
        sol['obj'] = self.model.getprimalobj(mosek.soltype.itr)
        return sol


if __name__ == "__main__":
    model_params = generate_feasible_test_case(
            num_paths=10,
            num_periods=10,
            num_prices=10,
            uncertainty_dim=1,
            seed=42
        )
    model = SOCP4LDR_Mosek(model_params, debug=True)
    model.build_model()
    model.solve(time_limit=600.0)
    solution = model.get_solution()
    print("Optimal X:", solution['X'])
