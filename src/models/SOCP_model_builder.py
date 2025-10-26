from functools import wraps
import logging
import time
from gurobipy import GRB, Model
import numpy as np

from src.config.config import Config
from src.models.model_builder import ModelBuilder, timeit_if_debug

class SOCPModelBuilder(ModelBuilder):
    def __init__(self, info="", solver="gurobi"):
        self.info = info
        self.solver = solver
        self.obj_val = 0
        self.solutions = {}
        self.INF =  100000000 # 100000000         #float('inf')
        if solver == "gurobi":
            # self.INF = GRB.INFINITY
            self.model = Model(info)
            self.model.setParam('TimeLimit', Config.default_time_limit)
            self.model.setParam('MIPGap', Config.default_mip_gap)
            self.model.setParam('Outputflag', Config.default_output_flag)

    """
    =======================================
    模型参数设置部分基类完成，子类可以共用
    """
    def set_model_params(self, model_params):
        """
        导入模型参数
        @params: model_params: dict 模型参数字典，包括：
            I1 : int
                不确定性维度 (ξ 的维数)。
            phi_list : list of hashable
                路径集合 Φ。
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
        self.model_params = model_params
        self.I1 = model_params['I1']
        self.phi_list = model_params["phi_list"]
        self.t_list = model_params['t_list']
        self.p_list = model_params['p_list']
        self.p_hat = model_params['p_hat']
        self.c_phi_t_p = model_params['c_phi_tp']
        self.t_d_phi = model_params['t_d_phi']

        # 不确定性参数
        self.set_uncertainty(
            mu=model_params["mu"],
            sigma_sq=model_params["sigma_sq"],
            Sigma=model_params["Sigma"]
        )

        # 网络参数
        self.set_network(
            paths=model_params["paths"],
            A_prime=model_params["A_prime"]
        )

         # 需求函数
        self.set_demand_function(
            d_0_phi_t=model_params["d_0_phi_t"],
            a=model_params["a"],
            d_z_phi_t_i=model_params["d_z_phi_t_i"]
        )

        self.set_Q_list()

        # 设置矩阵系数
        self.set_matrix()


    def set_uncertainty(self, mu, sigma_sq, Sigma):
        """
        设置不确定性参数：
          @params: mu : numpy array shape (I1,)           — E[z]
          @params: sigma_sq : numpy array shape (I1,)     — Var(z_i) or variance terms used in objective
          @params: Sigma : numpy array shape (I1,I1)      — 协方差矩阵 Cov(z)
        """
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.Sigma = Sigma
        self.cost_cov = float(np.ones(self.I1) @ self.Sigma @ np.ones(self.I1))

    def set_network(self, paths, A_prime):
        """
        网络信息：
          @params: paths : dict phi -> list of edges (edge as tuple (n,n'))
          @params: A_prime : dict edge -> capacity
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
                # if t <= self.t_d_phi[phi]:  # ← 只对需求期内的 t 添加约束
                    Q_list.append(('svc', phi, t))
                    Q_list.append(('mix', phi, t))
                    for p in self.p_list:
                        Q_list.append(('ng', phi, t, p))
                    Q_list.append(('nr', phi, t))
        self.Q_list = Q_list

        return Q_list


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
        d = np.zeros(3 * self.I1 + 3)
        h = np.zeros(3 * self.I1 + 3)

        # for i in range(self.I1):
        #     C[3*i+2, i] = -2.0
        #     D[3*i, i] = -1.0
        #     D[3*i+2, i] = -1.0
        #     h[3*i] = 1.0
        #     h[3*i+1] = -2 * self.mu[i]
        #     h[3*i+ 2] = -1.0
        # C[3*self.I1+1, :] = -2.0
        # d[3*self.I1] = -1.0
        # d[3*self.I1 + 2] = -1.0
        # h[3*self.I1] = 1.0
        # h[3*self.I1 +1] = -2 * sum(self.mu[i] for i in range(self.I1))
        # h[3*self.I1 + 2] = -1.0

        for i in range(self.I1):
            C[3*i, i] = -2.0
            D[3*i+1, i] = -1.0
            D[3*i+2, i] = -1.0
            h[3*i]   = -2.0 * self.mu[i]
            h[3*i+1] = -1.0
            h[3*i+2] =  1.0
        C[3*self.I1, :] = -2.0
        h[3*self.I1]   = -2.0 * float(np.sum(self.mu))
        h[3*self.I1+1] = -1.0
        h[3*self.I1+2] =  1.0
        d[3*self.I1+1] = -1.0
        d[3*self.I1+2] = -1.0


        E = - np.zeros((3 * self.I1 + 3, self.I1 + 1))
        self.C, self.D, self.d, self.h, self.E = C, D, d, h, E
        self.model_params['C'] = self.C
        self.model_params['D'] = self.D
        self.model_params['d'] = self.d
        self.model_params['E'] = self.E
        self.model_params['h'] = self.h

        # --- 调试输出 ---
        # print("\n--- DEBUG: Matrix Structure ---")
        # print("C matrix (shape: {}):".format(self.C.shape))
        # print(self.C)
        # print("D matrix (shape: {}):".format(self.D.shape))
        # print(self.D)
        # print("d vector:", self.d.shape)
        # print(self.d)
        # print("h vector:", self.h.shape)
        # print(self.h)
        # print("E matrix (shape: {}):".format(self.E.shape))
        # print(self.E)
        # print("--- END DEBUG ---\n")

    @timeit_if_debug
    def build_model(self):
        """ 构建模型 """
        try:
            self.set_parameters()

            self.create_variables()

            self.set_objective()

            self.add_constraints()

            self.update()

            self.print_model_info()
        except Exception as e:
            logging.error(f"构建模型时发生错误：{e}")
            raise e