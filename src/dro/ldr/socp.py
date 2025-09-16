import numpy as np
import gurobipy as gp
from gurobipy import GRB


class SOCP4LDR:
    def __init__(self, I1, phi_list, t_list, p_list, p_hat, c_phi_tp, t_d_phi) -> None:
        """
        Sets the parameters for the SOCP model.
        
        Parameters:
            I1: int
            mu, sigma_sq, Sigma: uncertainty set parameters
        """
        self.I1 = I1
        self.phi_list = phi_list
        self.t_list = t_list
        self.p_list = p_list
        self.p_hat = p_hat
        self.c_phi_tp = c_phi_tp
        self.t_d_phi = t_d_phi

    def set_uncertainty(self, mu, sigma_sq, Sigma):
        self.mu = mu
        self.sigma_sq = sigma_sq
        self.Sigma = Sigma

    def set_network(self, paths, A_prime):
        self.paths = paths
        self.A_prime = A_prime

    def set_demand_function(self, d_0_phi_t, a, d_z_phi_t_i):
        self.d_0_phi_t = d_0_phi_t
        self.d_z_phi_t_i = d_z_phi_t_i
        self.a = a

    def set_Q_list(self, Q_list):
        self.Q_list = Q_list


    def set_matrix(self):
        # --- Build matrices C, D, d, h ---
        C = np.zeros((2 * self.I1 + 2, self.I1))
        D = np.zeros((2 * self.I1 + 2, self.I1))

        for i in range(self.I1):
            C[2*i, i] = 2.0           # row 2*i (0-indexed) = 2e_i^T
            D[2*i+1, i] = 1.0      # row 2*i+1 = e_i^T

        C[2*self.I1, :] = 2.0      # last row of C: 2*1^T

        d = np.zeros(2 * self.I1 + 2)
        d[-1] = 1.0                     # d = [0,...,1]^T

        h = np.zeros(2 * self.I1 + 2)
        for i in range(self.I1):
            h[2*i] = 2 * self.mu[i]      # odd rows: 2*mu_i
            h[2*i+1] = 1.0                  # even rows: 1

        h[2*self.I1] = 2 * sum(self.mu[i] for i in range(self.I1))      # odd rows: 2*mu_i
        h[2*self.I1 + 1] = 1.0           # last: 1

        E = - np.eye(2 * self.I1 + 2)

        self.C, self.D, self.d, self.h, self.E = C, D, d, h, E


    def build_model(self):
        try:
            # --- Create Gurobi model ---
            self.model = gp.Model("DRO_Slot_Allocation_Complete_SOCP")
            self.model.Params.NonConvex = 2  # Required for SOC
            self.set_matrix()
            self.create_variables()
            self.add_constraints()
            self.set_objective()
        except Exception as e:
            print(f"Error building model: {e}")
        self.model.update()
        print("🔍 Model validation:")
        print(f"  Number of constraints: {self.model.numConstrs}")
        print(f"  Number of variables: {self.model.numVars}")
        print(f"  Number of SOC constraints: {len([c for c in self.model.getConstrs() if 'soc_' in c.ConstrName])}")


    def create_variables(self):
        # --- Decision Variables ---
        # Stage I: (X, Y)
        # \mathcal{X} = \left\{ (X, Y) \left| \sum_{\phi \in \Phi} (X_\phi + Y_\phi) \theta_{\phi nn'} \leq q_{nn'}, \quad \forall (n, n') \in \A; \quad X, Y \in \R^{|\Phi|}_+ \right. \right\}. \\
        self.X = self.model.addVars(self.phi_list, lb=0.0, ub=GRB.INFINITY, name="X")
        self.Y = self.model.addVars(self.phi_list, lb=0.0, ub=GRB.INFINITY, name="Y")

        # Stage II: dual variables
        self.r = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r")
        self.s = self.model.addVars(self.I1, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="s")
        self.t = self.model.addVars(self.I1, lb=-GRB.INFINITY, ub=0.0, name="t")   # t_k <= 0
        self.l = self.model.addVar(lb=-GRB.INFINITY, ub=0.0, name="l")        # l <= 0

        # G^0: probability allocation
        self.G0 = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                for p_val in self.p_list:
                    self.G0[(phi, t_val, p_val)] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"G0_{phi}_{t_val}_{p_val}"
                    )

        # G^{(z)}: linear coeff for z_i
        self.Gz = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                for p_val in self.p_list:
                    for i in range(self.I1):
                        self.Gz[(phi, t_val, p_val, i)] = self.model.addVar(
                            lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Gz_{phi}_{t_val}_{p_val}_{i}"
                        )

        # G^{(u)}: linear coeff for u_k
        self.Gu = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                for p_val in self.p_list:
                    for k in range(self.I1):
                        self.Gu[(phi, t_val, p_val, k)] = self.model.addVar(
                            lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Gu_{phi}_{t_val}_{p_val}_{k}"
                        )

        # R^0: base service cost
        self.R0 = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                self.R0[(phi, t_val)] = self.model.addVar(
                    lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"R0_{phi}_{t_val}"
                )

        # R^{(z)}: linear coeff for z_i in R
        self.Rz = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                for i in range(self.I1):
                    self.Rz[(phi, t_val, i)] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Rz_{phi}_{t_val}_{i}"
                    )

        # R^{(u)}: linear coeff for u_k in R
        self.Ru = {}
        for phi in self.phi_list:
            for t_val in self.t_list:
                for k in range(self.I1):
                    self.Ru[(phi, t_val, k)] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Ru_{phi}_{t_val}_{k}"
                    )

        # pi_q for each q in Q
        self.pi = {}
        for q_idx, q in enumerate(self.Q_list):
            self.pi[q] = self.model.addVars(2 * self.I1 + 2, lb=-GRB.INFINITY, name=f"pi_{q_idx}")

        # auxiliary variables
        self.delta0 = {}
        self.delta_z = {}
        self.delta_u = {}
        self.delta = {"0": self.delta0, "z": self.delta_z, "u": self.delta_u}
        for phi in self.phi_list:
            for t_val in self.t_list:
                self.delta0[(phi, t_val)] = self.model.addVar(
                    lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"delta0_{phi}_{t_val}"
                )
                for i in range(self.I1):
                    self.delta_z[(phi, t_val, i)] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"deltaz_{phi}_{t_val}_{i}"
                    )
                for k in range(self.I1):
                    self.delta_u[(phi, t_val, k)] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"deltau_{phi}_{t_val}_{k}"
                    )
        self.alpha0 = {}
        self.alpha_z = {}
        self.alpha_u = {}
        self.alpha = {"0": self.alpha0, "z": self.alpha_z, "u": self.alpha_u}
        for q in self.Q_list:
            self.alpha0[q] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"alpha0_{q}".replace(" ", "_")
                    )
            for i in range(self.I1):
                self.alpha_z[(q, i)] = self.model.addVar(
                            lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"alpha_z_{q}_{i}".replace(" ", "_")
                        )
            for k in range(self.I1):
                self.alpha_u[(q, k)] = self.model.addVar(
                            lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"alpha_u_{q}_{k}".replace(" ", "_")
                        )
        self.gamma = {}
        for q in self.Q_list:
            self.gamma[q] = self.model.addVar(
                        lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"gamma_{q}".replace(" ", "_")
                    )



    def set_objective(self):
        # --- Objective Function ---
        cost_cov = np.sum(self.Sigma)  # 1^T Sigma 1
        self.model.setObjective(
            self.r +
            sum(self.s[i] * self.mu[i] for i in range(self.I1)) +
            sum(self.t[i] * self.sigma_sq[i] for i in range(self.I1)) +
            self.l * cost_cov,
            GRB.MINIMIZE
        )

        # --- Constraint 1: Probability Allocation for G^0 ---
        for phi in self.phi_list:
            for t_val in self.t_list:
                expr = gp.LinExpr()
                for p_val in self.p_list:
                    expr += self.G0[(phi, t_val, p_val)]
                self.model.addConstr(expr == 1.0, name=f"prob_alloc_{phi}_{t_val}")


    def add_constraints(self):
        try:
            self.add_first_stage_constraints()
            self.set_delta_and_R()
            for q in self.Q_list:
                self.set_alpha_and_gamma(q)
                self.set_SOCP_block(q)
        except Exception as e:
            print(f"error in adding constraints: {e}")
    

    def add_first_stage_constraints(self):
        # & \mathcal{X} = \left\{ (X, Y) \left| \sum_{\phi \in \Phi} (X_\phi + Y_\phi) \theta_{\phi nn'} \leq q_{nn'}, \quad \forall (n, n') \in \A; \quad X, Y \in \R^{|\Phi|}_+ \right. \right\}. \\
        for edge, capacity in self.A_prime.items():
            n, n_prime = edge
            expr = gp.LinExpr()
            for phi in self.phi_list:
                if edge in self.paths.get(phi, []):
                    expr += self.X[phi] + self.Y[phi]
            self.model.addConstr(expr <= capacity, name=f"first_stage_{n}_{n_prime}")
            

    def set_delta_and_R(self):
        for phi in self.phi_list:
            for t in range(1, len(self.t_list)):
                t_val = self.t_list[t]
                # \Delta^0_{\phi t} = R^0_{\phi t} - Y_\phi + \sum_{t'=0}^{t-1} \left( d^0_{\phi t'} - a \sum_p p G^0_{\phi t' p} - a \hat{p}_\phi \right) \\
                self.model.addConstr(
                    self.delta0[(phi, t_val)] == self.R0[(phi, t_val)] - self.Y[phi] + sum(self.d_0_phi_t[(phi, t_prime)] - self.a * self.G0[(phi, t_prime, p_val)] - self.a * self.p_hat[phi] for t_prime in self.t_list[:t-1] for p_val in self.p_list), name=f"delta0_{phi}_{t_val}"
                )
                # \Delta^{(z)}_{\phi t, i} = R^{(z)}_{\phi t, i} + \sum_{t'=0}^{t-1} \left( d^{(z)}_{\phi t', i} - a \sum_p p G^{(z)}_{\phi t' p, i} \right) \\
                for i in range(self.I1):
                    self.model.addConstr(
                        self.delta_z[(phi, t_val, i)] == self.Rz[(phi, t_val, i)] + sum(self.d_z_phi_t_i[(phi, t_prime, i)] - self.a * self.Gz[(phi, t_prime, p_val, i)] for t_prime in self.t_list[:t-1] for p_val in self.p_list), name=f"deltaz_{phi}_{t_val}_{i}"
                    )
                # \Delta^{(u)}_{\phi t, k} = R^{(u)}_{\phi t, k} - a \sum_{t'=0}^{t-1} \sum_p p G^{(u)}_{\phi t' p, k}.
                for k in range(self.I1):
                    self.model.addConstr(
                        self.delta_u[(phi, t_val, k)] == self.Ru[(phi, t_val, k)] - self.a * sum(self.Gu[(phi, t_prime, p_val, k)] for t_prime in self.t_list[:t-1] for p_val in self.p_list), name=f"deltau_{phi}_{t_val}_{k}"
                    )

        for phi in self.phi_list:
            for t in range(1, len(self.t_list)):
                t_val = self.t_list[t]
                if t_val <= self.t_d_phi.get(phi, 0):
                    # \Delta^0_{\phi t} = 0, \quad 1 \forall \phi \in \Phi, \leq t \leq t_\phi(d) \\
                    self.model.addConstr(self.delta0[(phi, t_val)] == 0.0, name=f"delta00_{phi}_{t_val}")
                    # \Delta^{(z)}_{\phi t, i} = 0 \quad \forall \phi \in \Phi, \forall i, 1 \leq t \leq t_\phi(d) \\
                    for i in range(self.I1):
                        self.model.addConstr(self.delta_z[(phi, t_val, i)] == 0.0, name=f"deltaz0_{phi}_{t_val}_{i}")
                    # \Delta^{(u)}_{\phi t, k} = 0 \quad \forall \phi \in \Phi, \forall k=1,\dots,I_, 1 \leq t \leq t_\phi(d) \\
                    for k in range(self.I1):
                        self.model.addConstr(self.delta_u[(phi, t_val, k)] == 0.0, name=f"deltau0_{phi}_{t_val}_{k}")
                else:
                    # R^0_{\phi t} = 0 \quad \forall \phi \in \Phi, t \geq t_\phi(d)+1 \\
                    self.model.addConstr(self.R0[(phi, t_val)] == 0.0, name=f"R0_{phi}_{t_val}")
                    # R^{(z)}_{\phi t, i} = 0 \quad \forall \phi \in \Phi, \forall i, t \geq t_\phi(d)+1\\
                    for i in range(self.I1):
                        self.model.addConstr(self.Rz[(phi, t_val, i)] == 0.0, name=f"Rz_{phi}_{t_val}_{i}")
                    # R^{(u)}_{\phi t, k} = 0 \quad \forall \phi \in \Phi, \forall k=1,\dots,I_1, t \geq t_\phi(d)+1\\
                    for k in range(self.I1):
                        self.model.addConstr(self.Ru[(phi, t_val, k)] == 0.0, name=f"Ru_{phi}_{t_val}_{k}")


    def set_alpha_and_gamma(self, q):        
        # --- Constraint 2: Define alpha for each q type ---
        # For each q, compute alpha^{q,(z)}, alpha^{q,(u)}, alpha^q_0, gamma_q
        # These are not stored as vars, but computed from G,R,r,l 
        if q ==  'obj':
                # alpha0_obj = r - sum_{phi,t,p} c_{phi t p} G^0_{phi t p}
                alpha0_obj = self.r
                for phi in self.phi_list:
                    for t_val in self.t_list:
                        for p_val in self.p_list:
                            alpha0_obj -= self.c_phi_tp.get((phi, t_val, p_val), 0.0) * self.G0[(phi, t_val, p_val)]
                self.model.addConstr(self.alpha0[q] == alpha0_obj, name=f"alpha0_{q}")

                # alpha^{obj,(z)}_i = - sum_{phi,t,p} c_{phi t p} G^{(z)}_{phi t p, i}
                alpha_z_obj = [gp.LinExpr() for _ in range(self.I1)]
                for i in range(self.I1):
                    expr = gp.LinExpr()
                    for phi in self.phi_list:
                        for t_val in self.t_list:
                            for p_val in self.p_list:
                                expr -= self.c_phi_tp.get((phi, t_val, p_val), 0.0) * self.Gz[(phi, t_val, p_val, i)]
                    alpha_z_obj[i] = expr
                    self.model.addConstr(self.alpha_z[q, i] == alpha_z_obj[i], name=f"alpha_z_{q}_{i}")

                # alpha^{obj,(u)}_k = - sum_{phi,t,p} c_{phi t p} G^{(u)}_{phi t p, k}
                alpha_u_obj = [gp.LinExpr() for _ in range(self.I1)]
                for k in range(self.I1):
                    expr = gp.LinExpr()
                    for phi in self.phi_list:
                        for t_val in self.t_list:
                            for p_val in self.p_list:
                                expr -= self.c_phi_tp.get((phi, t_val, p_val), 0.0) * self.Gu[(phi, t_val, p_val, k)]
                    alpha_u_obj[k] = expr
                    self.model.addConstr(self.alpha_u[q, k] == alpha_u_obj[k], name=f"alpha_u_obj_{q}_{k}")

                # gamma_q = self.l  # for objective
                self.model.addConstr(self.gamma[q] == self.l, name=f"gamma_obj_{q}")
                            
                # self.alpha_z[q] = alpha_z_obj
                # self.alpha_u[q] = alpha_u_obj
                # self.alpha0[q] = alpha0_obj
                # self.gamma[q] = gamma_q

        elif isinstance(q, tuple) and q[0] == 'svc':
                phi, t_val = q[1], q[2]

                # alpha0_svc = d^0_{\phi t} - a \sum_p p G^0_{\phi t p} - R^0_{\phi t} + a \hat{p}_\phi 
                alpha0_svc = 0
                expr = self.d_0_phi_t.get((phi, t_val), 0.0)
                for p_val in self.p_list:
                    expr -= self.a * p_val * self.G0[(phi, t_val, p_val)]
                expr -= self.R0[(phi, t_val)]
                expr += self.a * self.p_hat[phi]
                alpha0_svc= expr
                self.model.addConstr(self.alpha0[q] == alpha0_svc, name=f"alpha0_svc_{phi}_{t_val}")


                # alpha^{svc,phi,t,(z)}_i = d^{(z)}_{phi t, i} - a * sum_p p * G^{(z)}_{phi t p, i} - R^{(z)}_{phi t, i}
                alpha_z_svc = [gp.LinExpr() for _ in range(self.I1)]
                for i in range(self.I1):
                    expr = self.d_z_phi_t_i.get((phi, t_val, i), 0.0)
                    for p_val in self.p_list:
                        expr -= self.a * p_val * self.Gz[(phi, t_val, p_val, i)]
                    expr -= self.Rz[(phi, t_val, i)]
                    alpha_z_svc[i] = expr
                    self.model.addConstr(self.alpha_z[q, i] == alpha_z_svc[i], name=f"alpha_z_svc_{phi}_{t_val}_{i}")

                # alpha^{svc,phi,t,(u)}_k = -a * sum_p p * G^{(u)}_{phi t p, k} - R^{(u)}_{phi t, k}
                alpha_u_svc = [gp.LinExpr() for _ in range(self.I1)]
                for k in range(self.I1):
                    expr = 0.0
                    for p_val in self.p_list:
                        expr -= self.a * p_val * self.Gu[(phi, t_val, p_val, k)]
                    expr -= self.Ru[(phi, t_val, k)]
                    alpha_u_svc[k] = expr
                    self.model.addConstr(self.alpha_u[q, k] == alpha_u_svc[k], name=f"alpha_u_svc_{phi}_{t_val}_{k}")

                # gamma_q = 0.0
                self.model.addConstr(self.gamma[q] == 0.0, name=f"gamma_svc_{phi}_{t_val}")

                # self.alpha_z[q]=alpha_z_svc
                # self.alpha_u[q] = alpha_u_svc
                # self.alpha0[q] = alpha0_svc
                # self.gamma[q] = gamma_q

        elif isinstance(q, tuple) and q[0] == 'mix':
                phi, t_val = q[1], q[2]

                # alpha0_mix = \sum_p G^0_{\phi t p} - 1
                alpha0_mix = gp.LinExpr()
                for p_val in self.p_list:
                    alpha0_mix += self.G0[(phi, t_val, p_val)]
                alpha0_mix -= 1
                self.model.addConstr(self.alpha0[q] == alpha0_mix, name=f"alpha0_mix_{phi}_{t_val}")


                # alpha^{mix,phi,t,(z)}_i = sum_p G^{(z)}_{phi t p, i}
                alpha_z_mix = [gp.LinExpr() for _ in range(self.I1)]
                for i in range(self.I1):
                    expr = gp.LinExpr()
                    for p_val in self.p_list:
                        expr += self.Gz[(phi, t_val, p_val, i)]
                    alpha_z_mix[i] = expr
                    self.model.addConstr(self.alpha_z[q, i] == alpha_z_mix[i], name=f"alpha_z_mix_{phi}_{t_val}_{i}")

                # alpha^{mix,phi,t,(u)}_k = sum_p G^{(u)}_{phi t p, k}
                alpha_u_mix = [gp.LinExpr() for _ in range(self.I1)]
                for k in range(self.I1):
                    expr = gp.LinExpr()
                    for p_val in self.p_list:
                        expr += self.Gu[(phi, t_val, p_val, k)]
                    alpha_u_mix[k] = expr
                    self.model.addConstr(self.alpha_u[q, k] == alpha_u_mix[k], name=f"alpha_u_mix_{phi}_{t_val}_{k}")

                # gamma_q = 0.0
                self.model.addConstr(self.gamma[q] == 0.0, name=f"gamma_mix_{phi}_{t_val}")

                # self.alpha_z[q] = alpha_z_mix
                # self.alpha_u[q] = alpha_u_mix
                # self.alpha0[q] = alpha0_mix
                # self.gamma[q] = gamma_q

        elif isinstance(q, tuple) and q[0] == 'ng':
                phi, t_val, p_val = q[1], q[2], q[3]

                # -G^0_{\phi t p}
                alpha0_ng = -self.G0[(phi, t_val, p_val)]
                self.model.addConstr(self.alpha0[q] == alpha0_ng, name=f"alpha0_ng_{phi}_{t_val}_{p_val}")

                # alpha^{ng,phi,t,p,(z)}_i = -G^{(z)}_{phi t p, i}
                alpha_z_ng = [gp.LinExpr() for _ in range(self.I1)]
                for i in range(self.I1):
                    alpha_z_ng[i] = -self.Gz[(phi, t_val, p_val, i)]
                    self.model.addConstr(self.alpha_z[q, i] == alpha_z_ng[i], name=f"alpha_z_ng_{phi}_{t_val}_{i}")

                # alpha^{ng,phi,t,p,(u)}_k = -G^{(u)}_{phi t p, k}
                alpha_u_ng = [gp.LinExpr() for _ in range(self.I1)]
                for k in range(self.I1):
                    alpha_u_ng[k] = -self.Gu[(phi, t_val, p_val, k)]
                    self.model.addConstr(self.alpha_u[q, k] == alpha_u_ng[k], name=f"alpha_u_ng_{phi}_{t_val}_{k}")

                # gamma_q = 0.0
                self.model.addConstr(self.gamma[q] == 0.0, name=f"gamma_{phi}_{t_val}")
                
                # self.alpha_z[q] = alpha_z_ng
                # self.alpha_u[q] = alpha_u_ng
                # self.alpha0[q] = alpha0_ng
                # self.gamma[q] = gamma_q

        elif isinstance(q, tuple) and q[0] == 'nr':
                phi, t_val = q[1], q[2]

                # alpha0_nr = (-R^0_{\phi t}
                alpha0_nr = -self.R0[(phi, t_val)]
                self.model.addConstr(self.alpha0[q] == alpha0_nr, name=f"alpha0_nr_{phi}_{t_val}")

                # alpha^{nr,phi,t,(z)}_i = -R^{(z)}_{phi t, i}
                alpha_z_nr = [gp.LinExpr() for _ in range(self.I1)]
                for i in range(self.I1):
                    alpha_z_nr[i] = -self.Rz[(phi, t_val, i)]
                    self.model.addConstr(self.alpha_z[q, i] == alpha_z_nr[i], name=f"alpha_z_nr_{phi}_{t_val}_{i}")

                # alpha^{nr,phi,t,(u)}_k = -R^{(u)}_{phi t, k}
                alpha_u_nr = [gp.LinExpr() for _ in range(self.I1)]
                for k in range(self.I1):
                    alpha_u_nr[k] = -self.Ru[(phi, t_val, k)]
                    self.model.addConstr(self.alpha_u[q, k] == alpha_u_nr[k], name=f"alpha_u_nr_{phi}_{t_val}_{k}")

                # gamma_q = 0.0
                self.model.addConstr(self.gamma[q] == 0.0, name=f"gamma_nr_{phi}_{t_val}")

                # self.alpha_z[q] = alpha_z_nr
                # self.alpha_u[q] = alpha_u_nr
                # self.alpha0[q] = alpha0_nr
                # self.gamma[q] = gamma_q

        else:
                raise ValueError(f"Unknown constraint type: {q}")

    def set_SOCP_block(self, q):
            # --- Dual Constraints: C^T pi_q = alpha^{q,(z)} ---
            # C[: , i]^T * pi_q = alpha^{q,(z)}_i
            for i in range(self.I1):
                expr = gp.LinExpr()
                for j in range(2 * self.I1 + 2):
                    expr += self.C[j, i] * self.pi[q][j]
                self.model.addConstr(expr == self.alpha_z[q, i], name=f"C_transpose_{q}_{i}".replace(" ", "_"))

            # --- D^T pi_q = alpha^{q,(u)} ---
            for k in range(self.I1):
                expr = gp.LinExpr()
                for j in range(2 * self.I1 + 2):
                    expr += self.D[j, k] * self.pi[q][j]
                self.model.addConstr(expr == self.alpha_u[q, k], name=f"D_transpose_{q}_{k}".replace(" ", "_"))

            # --- d^T pi_q = gamma_q ---
            expr = gp.LinExpr()
            for j in range(2 * self.I1 + 2):
                expr += self.d[j] * self.pi[q][j]
            self.model.addConstr(expr == self.gamma[q], name=f"d_transpose_{q}".replace(" ", "_"))

            # --- E^T pi_q = 0
            # Todo: check if this is correct
            for j in range(2 * self.I1 + 2):
                self.model.addConstr(self.E[j][j] * self.pi[q][j] == 0, name=f"E_transpose_{q}_{j}".replace(" ", "_"))

            # --- h^T pi_q <= -alpha0 ---
            expr = gp.LinExpr()
            for j in range(2*self.I1 + 2):
                expr += self.h[j] * self.pi[q][j]
            self.model.addConstr(expr <= -self.alpha0[q], name=f"h_transpose_{q}".replace(" ", "_"))

            # --- Second-order cone constraint: π_q ⪰_K 0
            pi_q = [self.pi[q][j] for j in range(2 * self.I1 + 2)]
            # (1) I1 independent 2D SOC constraints: || [pi_q[2i]] ||_2 <= pi_q[2i+1]
            for i in range(I1):
                norm_part: list[gp.Var] = [pi_q[2 * i]]  # Scalar inside norm
                upper_bound = pi_q[2 * i + 1]
                left_var = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"soc_single_{q}_{i}".replace(" ", "_"))
                self.model.addConstr(left_var == gp.norm(norm_part, 2), name=f"soc_single_norm_{q}_{i}".replace(" ", "_"))
                self.model.addConstr(left_var<=self.pi[q][2 * i + 1], name=f"soc_single_{q}_{i}".replace(" ", "_")) 
                self.model.addConstr(upper_bound >= 0, name=f"soc_single_nonneg_{q}_{i}".replace(" ", "_"))

            # (2) One (2)-dimensional SOC constraint: || [] pi_q[2I1-2]] ||_2 <= pi_q[2I1]
            norm_vars: list[gp.Var] = [pi_q[2 * self.I1]]
            upper_bound_var = pi_q[2 * I1 + 1]  # Last component
            left_var = self.model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"soc_agg_{q}".replace(" ", "_"))
            self.model.addConstr(left_var == gp.norm(norm_vars, 2), name=f"soc_agg_norm_{q}".replace(" ", "_"))
            self.model.addConstr(left_var <= upper_bound_var, name=f"soc_agg_{q}".replace(" ", "_"))
            self.model.addConstr(upper_bound_var >= 0, name=f"soc_agg_nonneg_{q}".replace(" ", "_"))


    def solve(self):
        # --- Optimize ---
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(f"✅ Optimal objective: {self.model.objVal:.6f}")
            return self.model.objVal, {
                'x': {phi: self.X[phi].X for phi in self.phi_list},
                'y': {phi: self.Y[phi].X for phi in self.phi_list},
                'r': self.r.X,
                's': [self.s[i].X for i in range(self.I1)],
                't': [self.t[i].X for i in range(self.I1)],
                'l': self.l.X,
                'G0': {(phi,t,p): self.G0[(phi,t,p)].X for phi in self.phi_list for t in self.t_list for p in self.p_list},
                'Gz': {(phi,t,p,i): self.Gz[(phi,t,p,i)].X for phi in self.phi_list for t in self.t_list for p in self.p_list for i in range(self.I1)},
                'Gu': {(phi,t,p,k): self.Gu[(phi,t,p,k)].X for phi in self.phi_list for t in self.t_list for p in self.p_list for k in range(self.I1)},
                'R0': {(phi,t): self.R0[(phi,t)].X for phi in self.phi_list for t in self.t_list},
                'Rz': {(phi,t,i): self.Rz[(phi,t,i)].X for phi in self.phi_list for t in self.t_list for i in range(self.I1)},
                'Ru': {(phi,t,k): self.Ru[(phi,t,k)].X for phi in self.phi_list for t in self.t_list for k in range(self.I1)},
                'pi': {q: [self.pi[q][j].X for j in range(3*self.I1+1)] for q in Q_list},
            }
        else:
            print(f"❌ Optimization failed with status {self.model.status}: {self.model.status}")
            self.model.computeIIS()
            self.model.write("model.ilp")
            self.model.write("model.lp")
            return None, None


# ================================
# 🚀 示例用法：模拟数据
# ================================

if __name__ == "__main__":
    

    # --- Problem Dimensions ---
    I1 = 2
    phi_list = ['P1']
    t_list = [1, 2]
    p_list = [1]
    p_hat = {'P1': 1.0}
    # --- Cost coefficients ---
    c_phi_tp = {
        ('P1', 1, 1): 100,
        ('P1', 2, 1): 90
    }
    t_d_phi = {
        'P1': 2
    }
     # phi_list,              # list of paths: e.g., ['P1','P2']
    # t_list,                  # list of time periods: e.g., [1,2,3]
    # p_list,                 # list of product types: e.g., [1,2]
    socp = SOCP4LDR(I1=I1, 
                    phi_list=phi_list, 
                    t_list=t_list, 
                    p_list=p_list, 
                    p_hat=p_hat, 
                    c_phi_tp=c_phi_tp, 
                    t_d_phi=t_d_phi)

    # --- Uncertainty parameters ---
    mu = np.array([0.5, 0.7])
    sigma_sq = np.array([0.1, 0.15])
    Sigma = np.array([[0.1, 0.02],
                      [0.02, 0.15]])
    socp.set_uncertainty(mu=mu, sigma_sq=sigma_sq, Sigma=Sigma)

    # --- network information ---
    paths = {'P1': [(1, 2), (2, 3)]}
    A_prime = {(1,2): 1, (2,3):1}
    socp.set_network(paths=paths, A_prime=A_prime)

    # --- Service parameters ---
    d_0_phi_t = {('P1', 0): 0.5, ('P1', 1): 0.6,
                   ('P1', 0): 0.4, ('P1', 1): 0.7}
    d_z_phi_t_i = {('P1', 1, 0): 0.5, ('P1', 1, 1): 0.6,
                   ('P1', 2, 0): 0.4, ('P1', 2, 1): 0.7}
    a = 1.0
    socp.set_demand_function(d_0_phi_t=d_0_phi_t, a=a, d_z_phi_t_i=d_z_phi_t_i)


    # --- Define Q: one obj, two svc, two mix, four ng, two nr ---
    Q_list = [
        'obj',
        ('svc', 'P1', 1), 
        ('svc', 'P1', 2),
        ('mix', 'P1', 1), 
        ('mix', 'P1', 2),
        ('ng', 'P1', 1, 1), 
        ('ng', 'P1', 1, 2),
        ('ng', 'P1', 2, 1), 
        ('ng', 'P1', 2, 2),
        ('nr', 'P1', 1), 
        ('nr', 'P1', 2)
    ]
    socp.set_Q_list(Q_list=Q_list)

    # --- Solve ---
    socp.build_model()

    socp.solve()