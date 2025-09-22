import numpy as np
import gurobipy as gp
from gurobipy import GRB

class SOCP4LDR:
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

    def __init__(self, I1, phi_list, t_list, p_list, p_hat, c_phi_tp, t_d_phi) -> None:
        """
        初始化并存储固定参数（不会被建为变量）。

        参数:
        ----------
        I1 : int
            不确定性维度 (ξ 的维数)。
        phi_list : list of hashable
            路径/产品集合 Φ。
        t_list : list of int
            时间段集合 T（通常从1开始; 若在模型中涉及 t=0, 请保证 d_0_phi_t 包含 t=0）。
        p_list : list of numeric
            价格/产品类型集合 P。
        p_hat : dict (phi -> base price)
            每条 phi 的基准价格 hat p_phi。
        c_phi_tp : dict ((phi,t,p) -> cost)
            成本系数 c_{φtp}（用于目标/obj α）。
        t_d_phi : dict (phi -> t_deadline)
            每个 φ 的需求有效期 t_φ(d)（用于 Δ / R 的 0/0 约束）。
        """
        self.I1 = I1
        self.phi_list = phi_list
        self.t_list = t_list
        self.p_list = p_list
        self.p_hat = p_hat
        self.c_phi_tp = c_phi_tp
        self.t_d_phi = t_d_phi

    # ---------------- Data setters ----------------
    def set_uncertainty(self, mu, sigma_sq, Sigma):
        """
        设置不确定性参数：
          mu : numpy array shape (I1,)           — E[z]
          sigma_sq : numpy array shape (I1,)     — Var(z_i) or variance terms used in objective
          Sigma : numpy array shape (I1,I1)      — 协方差矩阵 Cov(z)
        """
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.Sigma = Sigma

    def set_network(self, paths, A_prime):
        """
        网络信息：
          paths : dict phi -> list of edges (edge as tuple (n,n'))
          A_prime : dict edge -> capacity
        说明：用于第一阶段容量约束 Σ_{φ : edge∈path(φ)} (X_φ + Y_φ) ≤ capacity.
        """
        self.paths = paths
        self.A_prime = A_prime

    def set_demand_function(self, d_0_phi_t, a, d_z_phi_t_i):
        """
        需求函数相关参数：
          d_0_phi_t : dict (phi,t) -> scalar  (deterministic base demand d^0_{φt})
          a : scalar (price sensitivity)
          d_z_phi_t_i : dict (phi,t,i) -> scalar (coefficient for z_i in demand)
        说明：用于构造 Δ 表达式（见 set_delta_and_R）。
        """
        self.d_0_phi_t = d_0_phi_t
        self.d_z_phi_t_i = d_z_phi_t_i
        self.a = a

    def set_Q_list(self):
        """
        生成 Q_list（所有需要通过对偶/SOCP 处理的约束/目标的索引集合）。
        Q 包含：
          - 'obj'
          - ('svc', phi, t)
          - ('mix', phi, t)
          - ('ng', phi, t, p)
          - ('nr', phi, t)
        这些 q 会在 add_constraints 中逐一处理（先构造 alpha/gamma，再构造 SOCP 块）。
        """
        Q_list = []
        Q_list.append('obj')
        for phi in self.phi_list:
            for t in self.t_list:
                Q_list.append(('svc', phi, t))
                Q_list.append(('mix', phi, t))
                for p in self.p_list:
                    Q_list.append(('ng', phi, t, p))
                Q_list.append(('nr', phi, t))
        self.Q_list = Q_list

    def set_X_Y_value(self, X_value, Y_value):
        """
        设置 X, Y 的值（将一阶段变量固定为常数）。
        """
        for phi in self.phi_list:
            self.X[phi].LB = X_value[phi]
            self.X[phi].UB = X_value[phi]
            self.Y[phi].LB = Y_value[phi]
            self.Y[phi].UB = Y_value[phi]

    def set_matrix(self):
        """
        构造 C, D, d, h, E 矩阵（这些矩阵在论文中描述对偶/锥变换时出现）。
        - C, D: 根据论文中对 π 与 α,γ 的线性关系构造
        - d: 末项向量 (用于 d^T π = γ)
        - h: 支撑函数系数 (用于 h^T π ≤ -α_0)
        - E: 线性约束 (用于 E^T π = 0)
        注意：如果你的论文里 C/D/h 有不同定义，请替换此处构造。
        """
        C = np.zeros((3 * self.I1 + 3, self.I1))
        D = np.zeros((3 * self.I1 + 3, self.I1))

        for i in range(self.I1):
            C[3*i, i] = 2.0
            D[3*i+1, i] = 1.0
            D[3*i+2, i] = 1.0

        # 最后三行 C: [2 2 2,    d: [0
        #                        0 0 0,         1 
        #                        0 0 0]         1]
        C[3*self.I1, :] = 2.0

        d = np.zeros(3 * self.I1 + 3)
        d[3*self.I1 + 1] = 1.0
        d[3*self.I1 + 2] = 1.0

        # h 的构造常见于对偶支持函数：包含 mu 与常数项
        h = np.zeros(3 * self.I1 + 3)
        for i in range(self.I1):
            h[3*i] = 2 * self.mu[i]
            h[3*i+1] = 1.0
            h[3*i+2] = -1.0
        h[3*self.I1] = 2 * sum(self.mu[i] for i in range(self.I1))
        h[3*self.I1 + 1] = 1.0
        h[3*self.I1 + 2] = -1.0

        # E = [ [0, 0, -1] ... [0, 0 , -1] ] 
        E = - np.zeros((3 * self.I1 + 3, self.I1 + 1))
        for i in range(self.I1 + 1):
            E[3*i + 2, i] = -1

        self.C, self.D, self.d, self.h, self.E = C, D, d, h, E

        # --- 调试输出 ---
        print("\n--- DEBUG: Matrix Structure ---")
        print("C matrix (shape: {}):".format(C.shape))
        print(C)
        print("D matrix (shape: {}):".format(D.shape))
        print(D)
        print("d vector:", d.shape)
        print(d)
        print("h vector:", h.shape)
        print(h)
        print("E matrix (shape: {}):".format(E.shape))
        print(E)
        print("--- END DEBUG ---\n")

    # ---------------- Build & Variables ----------------
    def build_model(self):
        """
        整体建模入口：创建 Gurobi 模型对象，调用子函数建变量、添加约束、设目标。
        """
        self.model = gp.Model("DRO_Slot_Allocation_Complete_SOCP")
        # 设置参数
        self.model.Params.NonConvex = 2
        self.model.Params.OutputFlag = 1

        # 设置矩阵系数
        self.set_matrix()
        # 创建决策变量
        self.create_variables()
        # 添加约束
        self.add_constraints()
        # 设置目标函数
        self.set_objective()

        # update 并打印简单校验信息
        self.model.update()
        print("🔍 Model validation:")
        print(f"  Number of constraints: {self.model.numConstrs}")
        print(f"  Number of variables: {self.model.numVars}")

    def create_variables(self):
        """
        创建决策变量（仅变量创建，不添加约束）：
          - 第一阶段: X[φ], Y[φ] (>=0)
          - 第二阶段对偶: r, s_i, t_i (<=0), l (<=0)
          - 线性决策规则系数: G0[φ,t,p] (prob alloc coeff), Gz[φ,t,p,i], Gu[φ,t,p,k]
          - R 系数: R0[φ,t], Rz[φ,t,i], Ru[φ,t,k]
          - π_q: 对于每个 q, 创建 2*I1+2 个对偶分量 self.pi[q][0..2I1+1]
        说明/注意:
          - 我把 G0 的下界设为 0（概率/权重通常非负）。如果论文允许负数，请改回 lb=-inf。
          - 如果需要对 G0 施加 sum_p G0 == 1 或 <=1，请在 add_constraints 中添加（本文例子中未强制）。
        """
        # Stage I variables (nonnegative)  (非负，将被 set_X_Y_value 固定)
        self.X = self.model.addVars(self.phi_list, lb=0.0, name="X")
        self.Y = self.model.addVars(self.phi_list, lb=0.0, name="Y")

        # Stage II dual-like variables
        self.r = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r")
        self.s = self.model.addVars(self.I1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s")
        self.t = self.model.addVars(self.I1, lb=-GRB.INFINITY, ub=0.0, name="t")   # t_k <= 0
        self.l = self.model.addVar(lb=-GRB.INFINITY, ub=0.0, name="l")              # l <= 0

        # G^0 (prob weights) — 我这里设置 lb=0（概率/权重）
        self.G0 = {}
        self.Gz = {}
        self.Gu = {}
        for phi in self.phi_list:
            for t in self.t_list:
                for p in self.p_list:
                    # G0 非负建议 (概率权重)
                    self.G0[(phi, t, p)] = self.model.addVar(lb=0.0, name=f"G0_{phi}_{t}_{p}")
                    # Gz, Gu 表示 z,u 的线性系数，可以为实数
                    for i in range(self.I1):
                        self.Gz[(phi, t, p, i)] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Gz_{phi}_{t}_{p}_{i}")
                    for k in range(self.I1):
                        self.Gu[(phi, t, p, k)] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Gu_{phi}_{t}_{p}_{k}")

        # LDR: R^0, R^{(z)}, R^{(u)} (service cost base & coefficients)
        self.R0 = {}
        self.Rz = {}
        self.Ru = {}
        for phi in self.phi_list:
            for t in self.t_list:
                self.R0[(phi, t)] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"R0_{phi}_{t}")
                for i in range(self.I1):
                    self.Rz[(phi, t, i)] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Rz_{phi}_{t}_{i}")
                for k in range(self.I1):
                    self.Ru[(phi, t, k)] = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Ru_{phi}_{t}_{k}")

        # π_q variables: for each q an array of length 3*I1 + 3
        self.pi = {}
        for q in self.Q_list:
            # addVars returns a tupledict indexed by integers 0..(2I1+1)
            self.pi[q] = self.model.addVars(3 * self.I1 + 3, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"pi_{q}".replace(" ", "_"))

        # alpha, gamma, delta — **作为表达式容器**（LinExpr），不再建成 Vars
        self.alpha0 = {}   # alpha0[q] will be gp.LinExpr or Var-involving expr
        self.alpha_z = {}  # alpha_z[(q,i)]
        self.alpha_u = {}  # alpha_u[(q,k)]
        self.gamma = {}    # gamma[q]

        self.delta0 = {}   # delta0[(phi,t)] as LinExpr
        self.delta_z = {}  # delta_z[(phi,t,i)]
        self.delta_u = {}  # delta_u[(phi,t,k)]

    # ---------------- Objective ----------------
    def set_objective(self):
        """
        目标函数（对应论文中的对偶形式）：
          min r + sum_i s_i * mu_i + sum_i t_i * sigma_sq_i + l * (1^T Σ 1)
        """
        cost_cov = float(np.ones(self.I1) @ self.Sigma @ np.ones(self.I1))
        obj = self.r + gp.quicksum(self.s[i] * self.mu[i] for i in range(self.I1)) + gp.quicksum(self.t[i] * self.sigma_sq[i] for i in range(self.I1)) + self.l * cost_cov
        self.model.setObjective(
            obj,
            GRB.MINIMIZE
        )

    # ---------------- Constraints ----------------
    def add_constraints(self):
        """
        添加全体约束的主入口：
          - 第一阶段约束 (capacity)
          - Δ 与 R 之间的关系（表达式）及“0 条件”
          - 对每一个 q: 构造 α/γ 表达式并建立 SOCP 块（C^T π = α_z, D^T π = α_u, d^T π = γ, h^T π ≤ -α0）
          - 对每个 π_q 添加锥约束 (||..|| ≤ ..)（使用 addGenConstrNorm）
        """
        # # 1) 第一阶段约束
        # self.add_first_stage_constraints()
        
        # 2) delta & R 关系（构造表达式，添加 0/0 约束）
        self.set_delta_and_R()
        # 3) 对每个 q: 构造 alpha/gamma，并添加 SOCP 对偶约束块
        for q in self.Q_list:
            self.set_alpha_and_gamma(q)
            self.set_SOCP_block(q)

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
            self.model.addConstr(expr <= capacity, name=f"first_stage_{edge}")

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

                # 根据 t_d_phi 添加 Δ^*_{φt,*} == 0 或 R^*_{φt,*} == 0 的约束
                if t >= 1 and t <= self.t_d_phi.get(phi, 0):
                    # 在需求有效期内，带入仿射公式，Δ = 0（论文中的边界条件）
                    self.model.addConstr(self.delta0[(phi, t)] == 0.0, name=f"delta0_zero_{phi}_{t}")
                    for i in range(self.I1):
                        self.model.addConstr(self.delta_z[(phi, t, i)] == 0.0, name=f"deltaz_zero_{phi}_{t}_{i}")
                    for k in range(self.I1):
                        self.model.addConstr(self.delta_u[(phi, t, k)] == 0.0, name=f"deltau_zero_{phi}_{t}_{k}")
                # else:
                    # 超出有效期 => R 系数应为 0（论文中的另一端点条件）
                    self.model.addConstr(self.R0[(phi, t)] == 0.0, name=f"R0_zero_{phi}_{t}")
                    for i in range(self.I1):
                        self.model.addConstr(self.Rz[(phi, t, i)] == 0.0, name=f"Rz_zero_{phi}_{t}_{i}")
                    for k in range(self.I1):
                        self.model.addConstr(self.Ru[(phi, t, k)] == 0.0, name=f"Ru_zero_{phi}_{t}_{k}")

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
            for i in range(self.I1):
                self.alpha_z[(q, i)] = self.s[i] -gp.quicksum(
                    self.c_phi_tp.get((phi, t, p), 0.0) * self.Gz[(phi, t, p, i)]
                    for phi in self.phi_list for t in self.t_list for p in self.p_list
                )
            # α_u^{obj}_k = t - Σ c * Gu
            for k in range(self.I1):
                self.alpha_u[(q, k)] = self.t[k] -gp.quicksum(
                    self.c_phi_tp.get((phi, t, p), 0.0) * self.Gu[(phi, t, p, k)]
                    for phi in self.phi_list for t in self.t_list for p in self.p_list
                )
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
            for i in range(self.I1):
                self.alpha_z[(q, i)] = (self.d_z_phi_t_i.get((phi, t, i), 0.0)
                                         - self.a * gp.quicksum(p * self.Gz[(phi, t, p, i)] for p in self.p_list)
                                         - self.Rz[(phi, t, i)])
            # α_u^{svc}_k = - a Σ_p p Gu_{φtp,k} - Ru_{φt,k}
            for k in range(self.I1):
                self.alpha_u[(q, k)] = (- self.a * gp.quicksum(p * self.Gu[(phi, t, p, k)] for p in self.p_list)
                                         - self.Ru[(phi, t, k)])
            # γ_svc = 0
            self.gamma[q] = gp.LinExpr(0.0)

        # mix case
        elif isinstance(q, tuple) and q[0] == 'mix':
            phi, t = q[1], q[2]
            # α0_mix = Σ_p G0_{φtp} - 1
            self.alpha0[q] = gp.quicksum(self.G0[(phi, t, p)] for p in self.p_list) - 1.0
            for i in range(self.I1):
                self.alpha_z[(q, i)] = gp.quicksum(self.Gz[(phi, t, p, i)] for p in self.p_list)
            for k in range(self.I1):
                self.alpha_u[(q, k)] = gp.quicksum(self.Gu[(phi, t, p, k)] for p in self.p_list)
            self.gamma[q] = gp.LinExpr(0.0)

        # ng case (G nonneg)
        elif isinstance(q, tuple) and q[0] == 'ng':
            phi, t, p = q[1], q[2], q[3]
            self.alpha0[q] = - self.G0[(phi, t, p)]
            for i in range(self.I1):
                self.alpha_z[(q, i)] = - self.Gz[(phi, t, p, i)]
            for k in range(self.I1):
                self.alpha_u[(q, k)] = - self.Gu[(phi, t, p, k)]
            self.gamma[q] = gp.LinExpr(0.0)

        # nr case (R nonneg)
        elif isinstance(q, tuple) and q[0] == 'nr':
            phi, t = q[1], q[2]
            self.alpha0[q] = - self.R0[(phi, t)]
            for i in range(self.I1):
                self.alpha_z[(q, i)] = - self.Rz[(phi, t, i)]
            for k in range(self.I1):
                self.alpha_u[(q, k)] = - self.Ru[(phi, t, k)]
            self.gamma[q] = gp.LinExpr(0.0)

        else:
            raise ValueError(f"Unknown q type: {q}")

    def set_SOCP_block(self, q):
        """
        对每个 q 添加对偶线性等式与 SOCP 锥约束：
          - C^T π_q = α^{q,(z)}   (I1 条等式)
          - D^T π_q = α^{q,(u)}   (I1 条等式)
          - d^T π_q = γ_q              (1 条等式)
          - h^T π_q ≤ - α^q_0        (1 条不等式)
          - E^T π_q = 0                  (I1 + 1 条等式)
          - π_q ⪰_K 0                    (SOC锥约束：对每一对 (2i,2i+1) 以及最后一对 (2I1,2I1+1) 添加 Norm 约束)
        """
        # 1) z: C^T π_q = α_z
        for i in range(self.I1):
            lhs = gp.quicksum(self.C[j, i] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == self.alpha_z[(q, i)], name=f"Ctrans_q{q}_i{i}".replace(" ", "_"))

        # 2) u: D^T π_q = α_u
        for k in range(self.I1):
            lhs = gp.quicksum(self.D[j, k] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == self.alpha_u[(q, k)], name=f"Dtrans_q{q}_k{k}".replace(" ", "_"))

        # 3) u_{I1+1}: d^T π_q = γ_q
        lhs = gp.quicksum(self.d[j] * self.pi[q][j] for j in range(3 * self.I1 + 3))
        self.model.addConstr(lhs == self.gamma[q], name=f"dtrans_q{q}".replace(" ", "_"))

        # 4) v: E^T π_q = 0 : 对每对 (2i,2i+1) 使用 LinEq 约束，最后一对 (2I1,2I1+1) 也一样
        for i in range(self.I1 + 1):
            lhs = gp.quicksum(self.E[j, i] * self.pi[q][j] for j in range(3 * self.I1 + 3))
            self.model.addConstr(lhs == 0.0, name=f"Epi_q{q}_agg".replace(" ", "_"))

        # 5) h^T π_q <= - α0_q
        lhs = gp.quicksum(self.h[j] * self.pi[q][j] for j in range(3 * self.I1 + 3))
        self.model.addConstr(lhs <= - self.alpha0[q], name=f"htrans_q{q}".replace(" ", "_"))

        # 6) π_q ⪰_K 0 : 对每对 (3i, 3i+1) 使用 Norm 约束，最后一对 (3I1, 3I1+1) 也一样
        #   - 使用 addGenConstrNorm(RHSvar, list_of_vars, 2) 来添加 ||vars||_2 ≤ RHSvar
        #   - 为确保 RHSvar 非负（锥定义），显式添加 RHSvar ≥ 0
        #    每一对规范为: || [π_q[3i], π_q[3i+q]] ||_2 ≤ π_q[3i+2], 且 π_q[3i+2] ≥ 0
        for i in range(self.I1):
            norm_var = [self.pi[q][3 * i], self.pi[q][3 * i + 1]]               # 向量里的变量（这里是单个变量）
            rhs_var = self.pi[q][3 * i + 2]              # RHS 变量（必须是 Var）
            # RHS 需非负
            self.model.addConstr(rhs_var >= 0.0, name=f"pi_nonneg_q{q}_idx{2*i+1}".replace(" ", "_"))
            # ||norm_var||_2 <= rhs_var  (gen constr)
            self.model.addGenConstrNorm(rhs_var, norm_var, 2, name=f"norm_q{q}_pair{i}".replace(" ", "_"))

        # # aggregate last pair (index 3*I1, 3*I1+1)
        agg_norm_var = [self.pi[q][3 * self.I1], self.pi[q][3 * self.I1 + 1]]
        agg_rhs_var = self.pi[q][3 * self.I1 + 2]
        self.model.addConstr(agg_rhs_var >= 0.0, name=f"pi_nonneg_q{q}_last".replace(" ", "_"))
        self.model.addGenConstrNorm(agg_rhs_var, agg_norm_var, 2, name=f"norm_q{q}_agg".replace(" ", "_"))

    # ---------------- Solve & extract ----------------
    def solve(self, verbose=True):
        """
        优化并输出结果。
        """
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            if verbose:
                print(f"✅ Optimal objective (β(X,Y)): {self.model.objVal:.6f}")
            return True, self.model.objVal

        elif self.model.status == GRB.INF_OR_UNBD:
            if verbose:
                print("❌ Model is infeasible or unbounded.")
                # 计算不可行子系统 (IIS)
                self.model.computeIIS()
                print("\n🔍 Computing Irreducible Inconsistent Subsystem (IIS):")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"  🚫 Constraint '{c.ConstrName}' is in the IIS.")
                for gc in self.model.getGenConstrs():
                    if gc.IISGenConstr:
                        print(f"  🚫 GenConstraint '{gc.GenConstrName}' is in the IIS.")
                # 将 IIS 写入文件以便详细查看
                self.model.write("model_iis.ilp")
                print("  📄 IIS written to 'model_iis.ilp'")
            return False, None

        elif self.model.status == GRB.UNBOUNDED:
            if verbose:
                print("❌ Model unbounded")
            return False, None

        elif self.model.status == GRB.INFEASIBLE:
            if verbose:
                print("❌ Model infeasible")
                # 计算不可行子系统 (IIS)
                self.model.computeIIS()
                print("\n🔍 Computing Irreducible Inconsistent Subsystem (IIS):")
                for c in self.model.getConstrs():
                    if c.IISConstr:
                        print(f"  🚫 Constraint '{c.ConstrName}' is in the IIS.")
                for gc in self.model.getGenConstrs():
                    if gc.IISGenConstr:
                        print(f"  🚫 GenConstraint '{gc.GenConstrName}' is in the IIS.")
                self.model.write("model_iis.ilp")
                print("  📄 IIS written to 'model_iis.ilp'")
            return False, None

        else:
            if verbose:
                print(f"❌ Optimization terminated with status {self.model.status}")
            return False, None
        
    def get_solution(self):
        """
        返回关键变量的解（如果模型已求解且最优）。
        """
        if self.model.status != GRB.OPTIMAL:
            return None

        solution = {
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

        return solution


# ================= Example usage / 测试 =================
if __name__ == "__main__":
    # 示例数据（小规模，便于调试）
    # --- 1. 定义测试参数 ---
    I1 = 2  # 不确定性维度 (z1, z2)
    phi_list = ["P1", "P2"]  # 2 条路径
    t_list = [1, 2, 3]  # 3 个时间段
    p_list = [10.0, 15.0]  # 2 个价格点
    p_hat = {"P1": 5.0, "P2": 6.0}  # 每条路径的基准价格
    # 成本系数 c_{φtp}，假设成本随价格升高而增加
    c_phi_tp = {
        ("P1", 1, 10.0): 2.0, ("P1", 1, 15.0): 3.0,
        ("P1", 2, 10.0): 2.1, ("P1", 2, 15.0): 3.1, # 成本随时间略有上升
        ("P1", 3, 10.0): 2.2, ("P1", 3, 15.0): 3.2,
        ("P2", 1, 10.0): 1.8, ("P2", 1, 15.0): 2.8,
        ("P2", 2, 10.0): 1.9, ("P2", 2, 15.0): 2.9,
        ("P2", 3, 10.0): 2.0, ("P2", 3, 15.0): 3.0,
    }
    # 需求有效期：P1 在 t=2 后离港，P2 在 t=3 后离港
    t_d_phi = {"P1": 2, "P2": 3}

    # 不确定性参数
    mu = np.array([1.0, 0.4])  # E[z1]=1.0, E[z2]=0.4
    sigma_sq = np.array([1.0, 0.16])  # Var(z1)<=1.0, Var(z2)<=0.16
    # 协方差矩阵，假设 z1 和 z2 轻微正相关
    Sigma = np.array([[1.0, 0.2], [0.2, 0.16]])  # 1^T Σ 1 = 1.0 + 2*0.2 + 0.16 = 1.56

    # 网络参数 (占位，用于完整性)
    paths = {
        "P1": [("A", "B"), ("B", "C")],  # P1 经过 A->B, B->C
        "P2": [("A", "D"), ("D", "C")]   # P2 经过 A->D, D->C
    }
    A_prime = {
        ("A", "B"): 100.0,
        ("B", "C"): 80.0,
        ("A", "D"): 90.0,
        ("D", "C"): 85.0
    }

    # 需求参数
    # 基础需求 d0
    d_0_phi_t = {
        ("P1", 1): 8.0, ("P1", 2): 7.0, ("P1", 3): 0.0, # t=3 时 P1 已离港，需求为0
        ("P2", 1): 9.0, ("P2", 2): 8.5, ("P2", 3): 8.0
    }
    # 需求对 z 的敏感度 d(z)
    # 假设 z1 影响所有路径和时段，z2 主要影响 P2
    d_z_phi_t_i = {
        # P1 受 z1 影响
        ("P1", 1, 0): 2.0, ("P1", 2, 0): 1.8, ("P1", 3, 0): 0.0,
        ("P1", 1, 1): 0.0, ("P1", 2, 1): 0.0, ("P1", 3, 1): 0.0,
        # P2 受 z1 和 z2 影响
        ("P2", 1, 0): 2.2, ("P2", 2, 0): 2.0, ("P2", 3, 0): 1.8,
        ("P2", 1, 1): 1.0, ("P2", 2, 1): 0.9, ("P2", 3, 1): 0.8,
    }
    a = 1.0  # 价格敏感度

    # 第一阶段决策 (固定值)
    X_value = {"P1": 2.0, "P2": 3.0}  # 长协客户分配
    Y_value = {"P1": 5.0, "P2": 6.0}  # 临时客户预留


    socp = SOCP4LDR(I1=I1, phi_list=phi_list, t_list=t_list, p_list=p_list,
                    p_hat=p_hat, c_phi_tp=c_phi_tp, t_d_phi=t_d_phi)
    # 不确定性参数
    socp.set_uncertainty(mu=mu, sigma_sq=sigma_sq, Sigma=Sigma)

    # 网络
    socp.set_network(paths=paths, A_prime=A_prime)

    # 需求相关参数
    socp.set_demand_function(d_0_phi_t=d_0_phi_t, a=a, d_z_phi_t_i=d_z_phi_t_i)

    # Q 集合
    socp.set_Q_list()

    # Build & solve
    socp.build_model()

    # 设置 X 和 Y 的值
    socp.set_X_Y_value(X_value=X_value, Y_value=Y_value)

    # Solve the second stage problem
    socp.solve()
