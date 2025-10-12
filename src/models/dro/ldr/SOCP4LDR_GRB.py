import numpy as np
import os
import sys
import gurobipy as gp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_builder import MAXIMIZE, OPTIMAL, ModelBuilder, VType, timeit_if_debug


class SOCP4LDR(ModelBuilder):
    """
    二阶段分布鲁棒优化 (SOCP-based LDR)
    -------------------------------------------------------
    说明：
      - 用于求解给定第一阶段决策 X, Y 下的第二阶段问题 β(X, Y)。
      - 第一阶段决策: X_phi, Y_phi (非负)
      - 第二阶段通过 LDR 构建：G^0, G^{(z)}, G^{(u)}, R^0, R^{(z)}, R^{(u)}
      - 利用对偶与锥对偶转换，得到 π_q 等对偶变量，并构造 SOCP 块
      - 本实现把 alpha/gamma/delta 作为线性表达式 (LinExpr)，
        与论文的线性化/对偶关系一一对应（详见方法注释）
    """
    def __init__(self, model_params=None, debug=False) -> None:
        """
        初始化并存储固定参数（不会被建为变量）。

        参数:
        """
        super().__init__(info="SOCP4LDR", solver="gurobi")
        self.set_model_params(model_params)

    # ---------------- Build & Variables ----------------
    @timeit_if_debug
    def build_model(self):
        """
        整体建模入口：创建 Gurobi 模型对象，调用子函数建变量、添加约束、设目标。
        """
        # 设置参数
        self.model.Params.NonConvex = 2
        self.model.Params.OutputFlag = 1

        # 创建决策变量
        self.create_variables()
        # 添加约束
        self.add_constraints()
        # 设置目标函数
        self.set_objective()

        # 打印模型简单信息
        self.print_model_info()

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
        # Stage I variables (nonnegative)  (非负，将被 set_X_Y_value 固定)
        self.X = self.model.addVars(self.phi_list, lb=0.0, vtype=VType.CONTINUOUS, name="X")
        self.Y = self.model.addVars(self.phi_list, lb=0.0, vtype=VType.CONTINUOUS, name="Y")

        # Stage II dual-like variables
        self.r = self.model.addVar(lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name="r")
        self.s = self.model.addVars(self.I1, lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name="s")
        self.t = self.model.addVars(self.I1, lb=-self.INF , ub=0.0, vtype=VType.CONTINUOUS, name="t")              # t_k <= 0
        self.l = self.model.addVar(lb=-self.INF , ub=0.0, vtype=VType.CONTINUOUS, name="l")                                     # l <= 0

        # G^0 (prob weights)
        self.G0 = {}
        self.Gz = {}
        self.Gu = {}
        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    # G0
                    self.G0[(phi, t, p)] = self.model.addVar(lb=-self.INF, ub=self.INF,vtype=VType.CONTINUOUS, name=f"G0_{phi}_{t}_{p}")
                    # Gz, Gu 表示 z,u 的线性系数，可以为实数
                    for i in range(self.I1):
                        self.Gz[(phi, t, p, i)] = self.model.addVar(lb=-self.INF, ub=self.INF , vtype=VType.CONTINUOUS, name=f"Gz_{phi}_{t}_{p}_{i}")
                    for k in range(self.I1):
                        self.Gu[(phi, t, p, k)] = self.model.addVar(lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name=f"Gu_{phi}_{t}_{p}_{k}")

        # LDR: R^0, R^{(z)}, R^{(u)} (service cost base & coefficients)
        self.R0 = {}
        self.Rz = {}
        self.Ru = {}
        for phi in self.phi_list:
            for t in self.t_list:
                self.R0[(phi, t)] = self.model.addVar(lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name=f"R0_{phi}_{t}")
                for i in range(self.I1):
                    self.Rz[(phi, t, i)] = self.model.addVar(lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name=f"Rz_{phi}_{t}_{i}")
                for k in range(self.I1):
                    self.Ru[(phi, t, k)] = self.model.addVar(lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name=f"Ru_{phi}_{t}_{k}")

        # π_q variables: for each q an array of length 3*I1 + 3
        self.pi = {}
        for q in self.Q_list:
            # addVars returns a tupledict indexed by integers 0..(2I1+1)
            self.pi[q] = self.model.addVars(3 * self.I1 + 3, lb=-self.INF , ub=self.INF , vtype=VType.CONTINUOUS, name=f"pi_{q}".replace(" ", "_"))

        # alpha, gamma, delta — **作为表达式容器**（LinExpr），不再建成 Vars
        self.alpha0 = {}   # alpha0[q] will be gp.LinExpr or Var-involving expr
        self.alpha_z = {}  # alpha_z[(q,i)]
        self.alpha_u = {}  # alpha_u[(q,k)]
        self.gamma = {}    # gamma[q]

        self.delta0 = {}   # delta0[(phi,t)] as LinExpr
        self.delta_z = {}  # delta_z[(phi,t,i)]
        self.delta_u = {}  # delta_u[(phi,t,k)]

    # ---------------- Objective ----------------
    @timeit_if_debug
    def set_objective(self):
        """
        目标函数（对应论文中的对偶形式）：
          max:
            - Stage I:  Σ_φ p^hat_φ X_φ
            - Stage II:  r + sum_i s_i * mu_i + sum_i t_i * sigma_sq_i + l * (1^T Σ 1)
        """
        obj1 = gp.quicksum(self.p_hat[phi] * self.X[phi] for phi in self.phi_list)
        obj2 = self.r + gp.quicksum(self.s[i] * self.mu[i] for i in range(self.I1)) + gp.quicksum(self.t[i] * self.sigma_sq[i] for i in range(self.I1)) + self.l * self.cost_cov
        self.model.setObjective(
            obj1 + obj2,
            MAXIMIZE
        )

    # ---------------- Constraints ----------------
    @timeit_if_debug
    def add_constraints(self):
        """
        添加全体约束的主入口：
          - 第一阶段约束 (capacity)
          - Δ 与 R 之间的关系（表达式）及“0 条件”
          - 对每一个 q: 构造 α/γ 表达式并建立 SOCP 块
                （C^T π = α_z, D^T π = α_u, d^T π = γ, h^T π ≤ -α0, E^T π = 0）
          - 对每个 π_q 添加锥约束 (||x_1:n-1||2 ≤ x_n)
        """
        # # 1) 第一阶段约束
        self.add_first_stage_constraints()

        # 2) delta & R 关系（构造表达式，添加 0/0 约束）
        self.set_delta_and_R()
        # 3) 对每个 q: 构造 alpha/gamma，并添加 SOCP 对偶约束块
        for q in self.Q_list:
            self.add_SOCP_block(q)

        self.model.update()

    @timeit_if_debug
    def add_first_stage_constraints(self):
        """
        第一阶段网络容量约束：
          对于每条边 e=(n,n') in A_prime: Σ_{φ: e∈paths[φ]} (X_φ + Y_φ) ≤ capacity_e
        添加位置：模型中第一阶段约束（与论文中的可行集 X 一致）
        """
        for edge, capacity in self.A_prime.items():
            expr = gp.LinExpr()
            for phi in self.phi_list:
                if edge in self.paths.get(phi, []):
                    expr += self.X[phi] + self.Y[phi]
            self.model.addConstr(expr <= capacity, name=f"Capacity_{edge}".replace(" ", "_"))

    @timeit_if_debug
    def set_delta_and_R(self):
        """
        构造 Δ 表达式，并依据 t_d_phi 添加零/零互斥约束：
        论文形式（示例）：
          Δ^0_{φt}          = R^0_{φt} - Y_φ + Σ_{t'<t} ( d^0_{φt'} - a Σ_p p G^0_{φt'p} - a p̂_φ )
          Δ^{(z)}_{φt,i}  = R^{(z)}_{φt,i} + Σ_{t'<t} ( d^{(z)}_{φt',i} - a Σ_p p G^{(z)}_{φt'p,i} )
          Δ^{(u)}_{φt,k} = R^{(u)}_{φt,k} - a Σ_{t'<t} Σ_p p G^{(u)}_{φt'p,k}
        约束：
          如果 1 ≤t ≤ t_d_phi[φ]，则 Δ^*_{φt,*} == 0 （需求期内，递推公式带入LDR，这些量为0）
          否则 (t > t_d), R^*_{φt,*} == 0 （超出需求期，R 置 0）
        实现：
          - Δ 以 LinExpr 存储 self.delta0[(φ,t)] 等
          - 直接对 Δ 或 R 添加等式约束
        """
        for phi in self.phi_list:
            for t in self.t_list:
                # Δ^0 表达式
                expr0 = self.R0[(phi, t)] - self.Y[phi] + gp.quicksum(
                    (self.d_0_phi_t.get((phi, tp), 0.0)
                     - self.a * gp.quicksum(self.G0[(phi, tp, p)] * p for p in self.p_list)
                     - self.a * self.p_hat.get(phi, 0.0))
                    for tp in self.t_list if tp < t
                )
                self.delta0[(phi, t)] = expr0

                # Δ^{(z)} 表达式
                for i in range(self.I1):
                    exprz = self.Rz[(phi, t, i)] + gp.quicksum(
                        (self.d_z_phi_t_i.get((phi, tp, i), 0.0)
                         - self.a * gp.quicksum(self.Gz[(phi, tp, p, i)] * p for p in self.p_list))
                        for tp in self.t_list if tp < t
                    )
                    self.delta_z[(phi, t, i)] = exprz

                # Δ^{(u)} 表达式
                for k in range(self.I1):
                    expru = self.Ru[(phi, t, k)] - self.a * gp.quicksum(
                        self.Gu[(phi, tp, p, k)] * p for tp in self.t_list if tp < t for p in self.p_list
                    )
                    self.delta_u[(phi, t, k)] = expru

                # --- Step 2: 添加边界约束 ---
                t_deadline = self.t_d_phi.get(phi, 0)  # 获取该路径的需求截止时间

                # 根据 t_d_phi 添加 Δ^*_{φt,*} == 0 或 R^*_{φt,*} == 0 的约束
                if 1 <= t <= t_deadline:
                    # 在需求有效期内 (1 <= t <= t_d_phi)，Δ = 0
                    # print(f"  -> Adding Δ=0 constraints for (φ ={phi}, t={t})")
                    self.model.addConstr(self.delta0[(phi, t)] == 0.0, name=f"delta0_zero_{phi}_{t}")
                    for i in range(self.I1):
                        self.model.addConstr(self.delta_z[(phi, t, i)] == 0.0, name=f"delta_z_zero_{phi}_{t}_{i}")
                    for k in range(self.I1):
                        self.model.addConstr(self.delta_u[(phi, t, k)] == 0.0, name=f"delta_u_zero_{phi}_{t}_{k}")
                elif t > t_deadline:
                    # 超出有效期 (t > t_d_phi) => R 系数应为 0
                    # print(f"  -> Adding R=0 constraints for (φ ={phi}, t={t})")
                    self.model.addConstr(self.R0[(phi, t)] == 0.0, name=f"R0_zero_{phi}_{t}")
                    for i in range(self.I1):
                        self.model.addConstr(self.Rz[(phi, t, i)] == 0.0, name=f"Rz_zero_{phi}_{t}_{i}")
                    for k in range(self.I1):
                        self.model.addConstr(self.Ru[(phi, t, k)] == 0.0, name=f"Ru_zero_{phi}_{t}_{k}")
                else:
                    print(f"  -> No constraints added for ({phi}, {t})")
                    pass

    @timeit_if_debug
    def set_alpha_and_gamma(self, q):
        """
        对每个 q ∈ Q_list 构造 α^{q,(z)}, α^{q,(u)}, α^q_0, γ^q 的线性表达式 (LinExpr)。
        论文中这些 α/γ 是对偶线性组合的结果，我们在这里直接按章节公式逐项展开。

        q types (展开规则):
          - 'obj': 目标项 -> α0 = r - Σ_{φ,t,p} c_{φtp} G0_{φtp}
                    α^{(z)}_i = - Σ c_{φtp} Gz_{φtp,i}
                    α^{(u)}_k = - Σ c_{φtp} Gu_{φtp,k}
                    γ = l
          - ('svc', φ, t): 服务水平项
                    α0 = d^0_{φt} - a Σ_p p G0_{φtp} - R0_{φt} + a p̂_φ
                    α^{(z)}_i = d^{(z)}_{φt,i} - a Σ_p p Gz_{φtp,i} - Rz_{φt,i}
                    α^{(u)}_k = - a Σ_p p Gu_{φtp,k} - Ru_{φt,k}
                    γ = 0
          - ('mix', φ, t): 混合约束（概率和为1/或表述）
                    α0 = Σ_p G0_{φtp} - 1
                    α^{(z)}_i = Σ_p Gz_{φtp,i}
                    α^{(u)}_k = Σ_p Gu_{φtp,k}
                    γ = 0
          - ('ng', φ, t, p): G 非负相关（在论文里对应的 q）
                    α0 = - G0_{φtp}
                    α^{(z)}_i = - Gz_{φtp,i}
                    α^{(u)}_k = - Gu_{φtp,k}
                    γ = 0
          - ('nr', φ, t): R 非负相关
                    α0 = - R0_{φt}
                    α^{(z)}_i = - Rz_{φt,i}
                    α^{(u)}_k = - Ru_{φt,k}
                    γ = 0
        """
        # obj case
        if q == 'obj':
            # α0_obj = r - Σ c_{φtp} * G0_{φtp}
            self.alpha0[q] = self.r - gp.quicksum(
                self.c_phi_tp.get((phi, t, p), 0.0) * self.G0[(phi, t, p)]
                for phi in self.phi_list for t in self.t_list for p in self.p_list
            )
            # α_z^{obj}_i = s - Σ c * Gz
            self.alpha_z[q] = [self.s[i] -gp.quicksum(
                    self.c_phi_tp.get((phi, t, p), 0.0) * self.Gz[(phi, t, p, i)]
                    for phi in self.phi_list for t in self.t_list for p in self.p_list
                ) for i in range(self.I1)]
            # α_u^{obj}_k = t - Σ c * Gu
            self.alpha_u[q] = [self.t[k] -gp.quicksum(
                    self.c_phi_tp.get((phi, t, p), 0.0) * self.Gu[(phi, t, p, k)]
                    for phi in self.phi_list for t in self.t_list for p in self.p_list
                ) for k in range(self.I1)]
            # γ_obj = l
            self.gamma[q] = self.l

        # svc case
        elif isinstance(q, tuple) and q[0] == 'svc':
            phi, t = q[1], q[2]
            # α0_svc = d^0_{φt} - a Σ_p p G0_{φtp} - R0_{φt} + a p̂_φ
            self.alpha0[q] = (self.d_0_phi_t.get((phi, t), 0.0)
                              - self.a * gp.quicksum(p * self.G0[(phi, t, p)] for p in self.p_list)
                              - self.R0[(phi, t)]
                              + self.a * self.p_hat.get(phi, 0.0))
            # α_z^{svc}_i = d^{(z)}_{φt,i} - a Σ_p p Gz_{φtp,i} - Rz_{φt,i}
            self.alpha_z[q] = [(self.d_z_phi_t_i.get((phi, t, i), 0.0)
                                         - self.a * gp.quicksum(p * self.Gz[(phi, t, p, i)] for p in self.p_list)
                                         - self.Rz[(phi, t, i)]) for i in range(self.I1)]
            # α_u^{svc}_k = - a Σ_p p Gu_{φtp,k} - Ru_{φt,k}
            self.alpha_u[q] = [(- self.a * gp.quicksum(p * self.Gu[(phi, t, p, k)] for p in self.p_list)
                                         - self.Ru[(phi, t, k)]) for k in range(self.I1)]
            # γ_svc = 0
            self.gamma[q] = gp.LinExpr(0.0)

        # mix case
        elif isinstance(q, tuple) and q[0] == 'mix':
            phi, t = q[1], q[2]
            # α0_mix = Σ_p G0_{φtp} - 1
            self.alpha0[q] = gp.quicksum(self.G0[(phi, t, p)] for p in self.p_list) - 1.0
            self.alpha_z[q] = [gp.quicksum(self.Gz[(phi, t, p, i)] for p in self.p_list) for i in range(self.I1)]
            self.alpha_u[q] = [gp.quicksum(self.Gu[(phi, t, p, k)] for p in self.p_list) for k in range(self.I1)]
            self.gamma[q] = gp.LinExpr(0.0)

        # ng case (G nonneg)
        elif isinstance(q, tuple) and q[0] == 'ng':
            phi, t, p = q[1], q[2], q[3]
            self.alpha0[q] = - self.G0[(phi, t, p)]
            self.alpha_z[q] = [- self.Gz[(phi, t, p, i)] for i in range(self.I1)]
            self.alpha_u[q] = [- self.Gu[(phi, t, p, k)] for k in range(self.I1)]
            self.gamma[q] = gp.LinExpr(0.0)

        # nr case (R nonneg)
        elif isinstance(q, tuple) and q[0] == 'nr':
            phi, t = q[1], q[2]
            self.alpha0[q] = - self.R0[(phi, t)]
            self.alpha_z[q] = [- self.Rz[(phi, t, i)] for i in range(self.I1)]
            self.alpha_u[q] = [- self.Ru[(phi, t, k)] for k in range(self.I1)]
            self.gamma[q] = gp.LinExpr(0.0)

        else:
            raise ValueError(f"Unknown q type: {q}")

    @timeit_if_debug
    def add_SOCP_block(self, q):
        """
        对每个 q 添加对偶线性等式与 SOCP 锥约束：
          - C^T π_q = α^{q,(z)}   (I1 条等式)
          - D^T π_q = α^{q,(u)}   (I1 条等式)
          - d^T π_q = γ_q              (1 条等式)
          - h^T π_q ≤ - α^q_0        (1 条不等式)
          - E^T π_q = 0                  (I1 + 1 条等式)
          - π_q ⪰_K 0                    (SOC锥约束 I1+1 个三维二阶锥)
        """
        # 1) C^T π_q = α^{q,(z)}   (I1 条等式)
        # 2) D^T π_q = α^{q,(u)}   (I1 条等式)
        # 3) d^T π_q = γ_q              (1 条等式)
        # 4) E^T π_q = 0                  (I1 + 1 条等式)
        # 5) h^T π_q ≤ - α^q_0        (1 条不等式)
        self.set_dual_linear_constr(q)

        # 6) π_q ⪰_K 0 : 对每对 (3i, 3i+1) 使用 Norm 约束，最后一对 (3I1, 3I1+1) 也一样
        self.set_second_order_constr(q)

    @timeit_if_debug
    def set_dual_linear_constr(self, q):
        self.set_alpha_and_gamma(q)

        # 1) z: C^T π_q = α_z
        for i in range(self.I1):
            lhs = gp.quicksum(self.C[j, i] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == self.alpha_z[q][i], name=f"Ctrans_q{q}_i{i}".replace(" ", "_"))

        # 2) u: D^T π_q = α_u
        for k in range(self.I1):
            lhs = gp.quicksum(self.D[j, k] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == self.alpha_u[q][k], name=f"Dtrans_q{q}_k{k}".replace(" ", "_"))

        # 3) u_{I1+1}: d^T π_q = γ_q
        lhs = gp.quicksum(self.d[j] * self.pi[q][j] for j in range(3 * self.I1 + 3))
        self.model.addConstr(lhs == self.gamma[q], name=f"dtrans_q{q}".replace(" ", "_"))

        # 4) v: E^T π_q = 0 : 对每对 (2i,2i+1) 使用 LinEq 约束，最后一对 (2I1,2I1+1) 也一样
        for i in range(self.I1 + 1):
            lhs = gp.quicksum(self.E[j, i] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == 0.0, name=f"Epi_q{q}_agg_{i}".replace(" ", "_"))

        # 5) h^T π_q <= - α0_q
        lhs = gp.quicksum(self.h[j] * self.pi[q][j] for j in range(3 * self.I1 + 3))
        self.model.addConstr(lhs <= - self.alpha0[q], name=f"htrans_q{q}".replace(" ", "_"))

    @timeit_if_debug
    def set_second_order_constr(self, q):
        # 6) π_q ⪰_K 0 : I1+1个三维二阶锥，对 (3i, 3i+1, 3i+2) ⪰Q^3 0
        #    每一对规范为: || [π_q[3i], π_q[3i+1]] ||_2 ≤ π_q[3i+2], 且 π_q[3i+2] ≥ 0
        for i in range(self.I1):
            norm_var = [self.pi[q][3 * i], self.pi[q][3 * i + 1]]
            rhs_var = self.pi[q][3 * i + 2]
            # 锥约束: ||[π[3i], π[3i+1]]||_2 <= π[3i+2]
            self.model.addQConstr(gp.quicksum(v * v for v in norm_var) <= rhs_var * rhs_var, name=f"qconstr_norm_q{q}_pair{i}".replace(" ", "_"))

        # 聚合项的锥约束: || [π_q[3I1], π_q[3I1+1]] ||_2 ≤ π_q[3I1+2]
        agg_norm_var = [self.pi[q][3 * self.I1], self.pi[q][3 * self.I1 + 1]]
        agg_rhs_var = self.pi[q][3 * self.I1 + 2]
        self.model.addQConstr(gp.quicksum(v * v for v in agg_norm_var) <= agg_rhs_var * agg_rhs_var, name=f"qconstr_norm_q{q}_agg".replace(" ", "_"))

    # ---------------- Solve & extract ----------------
    @timeit_if_debug
    def solve(self, verbose=True):
        """
        优化并输出结果。
        """
        self.model.Params.MIPGap = 0.01  # 1% 的 Gap
        self.model.Params.TimeLimit = 300  # 60 秒的时间限制

        self.model.optimize()

        self.print_model_status()

        self.exact_solution()

        self.print_solution()


    def get_solution(self):
        """
        返回关键变量的解（如果模型已求解且最优）。
        """
        if self.model.status != OPTIMAL:
            return None

        return self.exact_solution()

    def exact_solution(self):
        self.solutions = {
            'obj_val': self.model.objVal,
            'solve_time': self.model.Runtime,
            # 'Gap': self.model.MIPGap,
            'X': {phi: self.X[phi].X for phi in self.phi_list},
            'Y': {phi: self.Y[phi].X for phi in self.phi_list},
            'r': self.r.X,
            's': [self.s[i].X for i in range(self.I1)],
            't': [self.t[k].X for k in range(self.I1)],
            'l': self.l.X,
            'G0': {(phi, t, p): self.G0[(phi, t, p)].X for phi in self.phi_list for t in self.t_list for p in self.p_list},
            'Gz': {(phi, t, p, i): self.Gz[(phi, t, p, i)].X for phi in self.phi_list for t in self.t_list for p in self.p_list for i in range(self.I1)},
            'Gu': {(phi, t, p, k): self.Gu[(phi, t, p, k)].X for phi in self.phi_list for t in self.t_list for p in self.p_list for k in range(self.I1)},
            'R0': {(phi, t): self.R0[(phi, t)].X for phi in self.phi_list for t in self.t_list},
            'Rz': {(phi, t, i): self.Rz[(phi, t, i)].X for phi in self.phi_list for t in self.t_list for i in range(self.I1)},
            'Ru': {(phi, t, k): self.Ru[(phi, t, k)].X for phi in self.phi_list for t in self.t_list for k in range(self.I1)},
        }

        self.solutions['pi'] = self.get_pi_solution()
        self.solutions['alpha'] = self.extract_alpha_values(self.model_params)

        return self.solutions

    def get_pi_solution(self):
        """
        返回对偶变量的解（如果模型已求解且最优）。
        """
        if self.model.status != OPTIMAL:
            return None

        pi_solution = {
                    q: [self.pi[q][i].x for i in range(3 * self.I1 + 3)]
                    for q in self.Q_list
        }

        return pi_solution

    def extract_alpha_values(self, model_params):
        """
        根据模型的最优解，提取并计算每个 q ∈ Q 对应的仿射系数 (alpha0, alpha_z, alpha_u, gamma) 的数值。

        这个函数严格遵循 `model.pdf` 附录 D.2(ii) 的公式 (A61a-d)。

        Parameters:
        - solution: 模型解字典，包含 'G0', 'Gz', 'Gu', 'R0', 'Rz', 'Ru' 等LDR系数的最优值。
        - model_params: 模型参数字典，必须包含以下键:
            * 'phi_list', 't_list', 'p_list'
            * 'c_phi_tp': 成本字典
            * 'd_0_phi_t': 基础需求字典
            * 'd_z_phi_t_i': 需求对 z 的敏感度字典
            * 'p_hat': 基准价格字典
            * 'a': 价格敏感度

        Returns:
        - alpha_values: 一个字典，其键为 q ∈ Q，值为另一个字典 {'alpha0': ..., 'alpha_z': [...], 'alpha_u': [...], 'gamma': ...}
        """
        # 从参数中解包
        phi_list = model_params['phi_list']
        t_list = model_params['t_list']
        p_list = model_params['p_list']
        c_phi_tp = model_params['c_phi_tp']
        d_0_phi_t = model_params['d_0_phi_t']
        d_z_phi_t_i = model_params['d_z_phi_t_i']
        p_hat = model_params['p_hat']
        a = model_params['a']
        I1 = model_params['I1']

        # 初始化返回字典
        alpha_values = {}

        # 1. 处理 'obj' 类型
        q = 'obj'
        alpha0_val = self.solutions['r'] - sum(
            c_phi_tp.get((phi, t, p), 0.0) * self.solutions['G0'][(phi, t, p)]
            for phi in phi_list for t in t_list for p in p_list
        )
        alpha_z_vals = []
        alpha_u_vals = []
        for i in range(I1):
            alpha_z_i = self.solutions['s'][i] - sum(
                c_phi_tp.get((phi, t, p), 0.0) * self.solutions['Gz'][(phi, t, p, i)]
                for phi in phi_list for t in t_list for p in p_list
            )
            alpha_z_vals.append(alpha_z_i)
        for k in range(I1):
            alpha_u_k = self.solutions['t'][k] - sum(
                c_phi_tp.get((phi, t, p), 0.0) * self.solutions['Gu'][(phi, t, p, k)]
                for phi in phi_list for t in t_list for p in p_list
            )
            alpha_u_vals.append(alpha_u_k)
        gamma_val = self.solutions['l']

        alpha_values[q] = {
            'alpha0': alpha0_val,
            'alpha_z': alpha_z_vals,
            'alpha_u': alpha_u_vals,
            'gamma': gamma_val
        }

        # 2. 处理其他类型的 q
        for phi in phi_list:
            for t in t_list:
                # (a) 'svc' - 服务水平约束
                q = ('svc', phi, t)
                alpha0_val = (d_0_phi_t.get((phi, t), 0.0)
                            - a * sum(p * self.solutions['G0'][(phi, t, p)] for p in p_list)
                            - self.solutions['R0'][(phi, t)]
                            + a * p_hat.get(phi, 0.0))
                alpha_z_vals = []
                alpha_u_vals = []
                for i in range(I1):
                    alpha_z_i = (d_z_phi_t_i.get((phi, t, i), 0.0)
                                - a * sum(p * self.solutions['Gz'][(phi, t, p, i)] for p in p_list)
                                - self.solutions['Rz'][(phi, t, i)])
                    alpha_z_vals.append(alpha_z_i)
                for k in range(I1):
                    alpha_u_k = (-a * sum(p * self.solutions['Gu'][(phi, t, p, k)] for p in p_list)
                                - self.solutions['Ru'][(phi, t, k)])
                    alpha_u_vals.append(alpha_u_k)
                gamma_val = 0.0

                alpha_values[q] = {
                    'alpha0': alpha0_val,
                    'alpha_z': alpha_z_vals,
                    'alpha_u': alpha_u_vals,
                    'gamma': gamma_val
                }

                # (b) 'mix' - 混合约束 (概率和)
                q = ('mix', phi, t)
                alpha0_val = sum(self.solutions['G0'][(phi, t, p)] for p in p_list) - 1.0
                alpha_z_vals = [sum(self.solutions['Gz'][(phi, t, p, i)] for p in p_list) for i in range(I1)]
                alpha_u_vals = [sum(self.solutions['Gu'][(phi, t, p, k)] for p in p_list) for k in range(I1)]
                gamma_val = 0.0

                alpha_values[q] = {
                    'alpha0': alpha0_val,
                    'alpha_z': alpha_z_vals,
                    'alpha_u': alpha_u_vals,
                    'gamma': gamma_val
                }

                # (c) 'ng' - G 非负约束
                for p in p_list:
                    q = ('ng', phi, t, p)
                    alpha0_val = -self.solutions['G0'][(phi, t, p)]
                    alpha_z_vals = [-self.solutions['Gz'][(phi, t, p, i)] for i in range(I1)]
                    alpha_u_vals = [-self.solutions['Gu'][(phi, t, p, k)] for k in range(I1)]
                    gamma_val = 0.0

                    alpha_values[q] = {
                        'alpha0': alpha0_val,
                        'alpha_z': alpha_z_vals,
                        'alpha_u': alpha_u_vals,
                        'gamma': gamma_val
                    }

                # (d) 'nr' - R 非负约束
                q = ('nr', phi, t)
                alpha0_val = -self.solutions['R0'][(phi, t)]
                alpha_z_vals = [-self.solutions['Rz'][(phi, t, i)] for i in range(I1)]
                alpha_u_vals = [-self.solutions['Ru'][(phi, t, k)] for k in range(I1)]
                gamma_val = 0.0

                alpha_values[q] = {
                    'alpha0': alpha0_val,
                    'alpha_z': alpha_z_vals,
                    'alpha_u': alpha_u_vals,
                    'gamma': gamma_val
                }

        return alpha_values

    def get_alpha_solution(self):
        """
        返回对偶变量的解（如果模型已求解且最优）。
        """
        if self.model.status != OPTIMAL:
            return None

        return {}

    def print_solution(self):
        """
        打印关键变量的解（如果模型已求解且最优）。
        """
        solutions = self.get_solution()
        if solutions is None:
            print("No solution found.")
        else:
            print("Solution:")
            for key, value in solutions.items():
                print(f"{key}: {value}")
