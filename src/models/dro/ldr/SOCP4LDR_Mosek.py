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
        super().__init__(info="SOCP4LDR", solver="mosek")
        self.debug = debug
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
        self._add_first_stage_capacity()
        self._add_delta_and_R_constraints()
        self._add_SOCP_blocks()

    def _add_first_stage_capacity(self):
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

    def _add_delta_and_R_constraints(self):
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

        # (26): t_k <= 0, l <= 0
        for k in range(self.I1):
            self.model.putvarbound(self.t_idx[k], mosek.boundkey.up, -0.0, 0.0)
        self.model.putvarbound(self.l_idx, mosek.boundkey.up, -0.0, 0.0)

        # G0 >= 0
        for key in self.G0_idx:
            self.model.putvarbound(self.G0_idx[key], mosek.boundkey.lo, 0.0, +0.0)

    def _add_SOCP_blocks(self):
        """约束 (20)–(25) for each q"""
        I1 = self.I1
        for q in self.Q_list:
            alpha0, alpha_z, alpha_u, gamma = self._build_alpha_gamma(q)

            # (20): C^T pi_q = alpha_z
            for i in range(I1):
                vars_ = self.pi_idx[q] + [alpha_z[i]]
                coeffs = self.C[:, i].tolist() + [-1.0]
                self.model.appendcons(1)
                con_idx = self.model.getnumcon() - 1
                self.model.putarow(con_idx, vars_, coeffs)
                self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

            # (21): D^T pi_q = alpha_u
            for k in range(I1):
                vars_ = self.pi_idx[q] + [alpha_u[k]]
                coeffs = self.D[:, k].tolist() + [-1.0]
                self.model.appendcons(1)
                con_idx = self.model.getnumcon() - 1
                self.model.putarow(con_idx, vars_, coeffs)
                self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

            # (22): d^T pi_q = gamma
            vars_ = self.pi_idx[q] + [gamma]
            coeffs = self.d.tolist() + [-1.0]
            self.model.appendcons(1)
            con_idx = self.model.getnumcon() - 1
            self.model.putarow(con_idx, vars_, coeffs)
            self.model.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)

            # (24): h^T pi_q + alpha0 <= 0
            vars_ = self.pi_idx[q] + [alpha0]
            coeffs = self.h.tolist() + [1.0]
            self.model.appendcons(1)
            con_idx = self.model.getnumcon() - 1
            self.model.putarow(con_idx, vars_, coeffs)
            self.model.putconbound(con_idx, mosek.boundkey.up, -0.0, 0.0)

            # (25): SOC constraints — I1+1 个 3D 锥
            for i in range(I1):
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
            self.model.putafefentry(afe_idx, self.pi_idx[q][3*I1+2], -1.0)
            self.model.putafeg(afe_idx, 0.0)
            self.model.putafefentry(afe_idx+1, self.pi_idx[q][3*I1], -1.0)
            self.model.putafeg(afe_idx+1, 0.0)
            self.model.putafefentry(afe_idx+2, self.pi_idx[q][3*I1+1], -1.0)
            self.model.putafeg(afe_idx+2, 0.0)
            quad_dom = self.model.appendquadraticconedomain(3)
            self.model.appendacc(quad_dom, [afe_idx, afe_idx+1, afe_idx+2], None)

    def _build_alpha_gamma(self, q):
        """返回 alpha0, alpha_z (list), alpha_u (list), gamma 的变量索引"""
        I1 = self.I1
        if q == 'obj':
            # alpha0 = r - sum c G0
            alpha0 = self.r_idx
            alpha_z = [self.s_idx[i] for i in range(I1)]
            alpha_u = [self.t_idx[k] for k in range(I1)]
            gamma = self.l_idx
        else:
            typ = q[0]
            if typ == 'svc':
                phi, t = q[1], q[2]
                alpha0 = self._add_linear_expr([
                    (1.0, self.d_0_phi_t.get((phi, t), 0.0)),
                    (-self.a, [self.G0_idx[(phi, t, p)] for p in self.p_list], self.p_list),
                    (-1.0, self.R0_idx[(phi, t)]),
                    (1.0, self.a * self.p_hat.get(phi, 0.0))
                ])
                alpha_z = []
                for i in range(I1):
                    expr = self._add_linear_expr([
                        (1.0, self.d_z_phi_t_i.get((phi, t, i), 0.0)),
                        (-self.a, [self.Gz_idx[(phi, t, p, i)] for p in self.p_list], self.p_list),
                        (-1.0, self.Rz_idx[(phi, t, i)])
                    ])
                    alpha_z.append(expr)
                alpha_u = []
                for k in range(I1):
                    expr = self._add_linear_expr([
                        (-self.a, [self.Gu_idx[(phi, t, p, k)] for p in self.p_list], self.p_list),
                        (-1.0, self.Ru_idx[(phi, t, k)])
                    ])
                    alpha_u.append(expr)
                gamma = self._add_const_var(0.0)
            elif typ == 'mix':
                phi, t = q[1], q[2]
                alpha0 = self._add_linear_expr([
                    (1.0, [self.G0_idx[(phi, t, p)] for p in self.p_list], [1.0]*len(self.p_list)),
                    (-1.0, 1.0)
                ])
                alpha_z = [self._add_linear_expr([(1.0, [self.Gz_idx[(phi, t, p, i)] for p in self.p_list], [1.0]*len(self.p_list))]) for i in range(I1)]
                alpha_u = [self._add_linear_expr([(1.0, [self.Gu_idx[(phi, t, p, k)] for p in self.p_list], [1.0]*len(self.p_list))]) for k in range(I1)]
                gamma = self._add_const_var(0.0)
            elif typ == 'ng':
                phi, t, p = q[1], q[2], q[3]
                alpha0 = self._add_linear_expr([(-1.0, self.G0_idx[(phi, t, p)])])
                alpha_z = [self._add_linear_expr([(-1.0, self.Gz_idx[(phi, t, p, i)])]) for i in range(I1)]
                alpha_u = [self._add_linear_expr([(-1.0, self.Gu_idx[(phi, t, p, k)])]) for k in range(I1)]
                gamma = self._add_const_var(0.0)
            elif typ == 'nr':
                phi, t = q[1], q[2]
                alpha0 = self._add_linear_expr([(-1.0, self.R0_idx[(phi, t)])])
                alpha_z = [self._add_linear_expr([(-1.0, self.Rz_idx[(phi, t, i)])]) for i in range(I1)]
                alpha_u = [self._add_linear_expr([(-1.0, self.Ru_idx[(phi, t, k)])]) for k in range(I1)]
                gamma = self._add_const_var(0.0)
            else:
                raise ValueError(f"Unknown q: {q}")
        return alpha0, alpha_z, alpha_u, gamma

    def _add_const_var(self, val: float) -> int:
        """添加一个固定值为 val 的变量（通过约束 x = val）"""
        idx = self._add_var(f"const_{len(self._var_index)}")
        self.model.appendvars(1)
        self.model.putcj(idx, 0.0)
        self.model.putvarbound(idx, mosek.boundkey.fx, val, val)
        return idx

    def _add_linear_expr(self, terms):
        """terms: list of (coeff, var_or_val, [multipliers])"""
        idx = self._add_var(f"expr_{len(self._var_index)}")
        self.model.appendvars(1)
        self.model.putcj(idx, 0.0)
        self.model.putvarbound(idx, mosek.boundkey.fr, -0.0, +0.0)

        vars_ = [idx]
        coeffs = [-1.0]
        rhs = 0.0
        for term in terms:
            coeff = term[0]
            if len(term) == 2:
                val = term[1]
                if isinstance(val, (int, float)):
                    rhs += coeff * val
                else:
                    vars_.append(val)
                    coeffs.append(coeff)
            elif len(term) == 3:
                var_list, mul_list = term[1], term[2]
                for v, m in zip(var_list, mul_list):
                    vars_.append(v)
                    coeffs.append(coeff * m)
        self.model.appendcons(1)
        con_idx = self.model.getnumcon() - 1
        self.model.putarow(con_idx, vars_, coeffs)
        self.model.putconbound(con_idx, mosek.boundkey.fx, rhs, rhs)
        return idx

    def solve(self, time_limit: float = 300.0, rel_gap: float = 1e-3, verbose: bool = True):
        self.model.putdouparam(mosek.dparam.intpnt_co_tol_rel_gap, rel_gap)
        self.model.putdouparam(mosek.dparam.optimizer_max_time, time_limit)
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
