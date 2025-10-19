import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Any, Tuple
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
print(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model_builder import OPTIMAL, ModelBuilder, VType, timeit_if_debug
from src.utils.model_params import generate_feasible_test_case


class DeterministicModel(ModelBuilder):
    """
    确定性仓位分配与定价问题模型 (Deterministic Slot Allocation and Pricing Problem)

    该类实现了 Manuscript_XX_20250904 -LJ.pdf 中公式 (1)-(7) 描述的确定性模型。
    它是一个混合整数线性规划 (MILP) 模型，但默认使用其 LP 松弛形式 (Gϕtp ∈ [0, 1])，
    以便与分布鲁棒优化 (DRO) 模型进行比较。
    """

    def __init__(self, model_params=None, use_lp_relaxation=True) -> None:
        """
        @param model_params: 模型参数字典
        """
        super().__init__(info="DM", solver="gurobi")
        self.set_model_params(model_params)
        self.use_lp_relaxation = use_lp_relaxation

    @timeit_if_debug
    def create_variables(self):
        """创建模型变量。"""
        # Stage I variables (nonnegative)  (非负)
        self.X = self.model.addVars(self.phi_list, lb=0.0, vtype=VType.CONTINUOUS, name="X")
        self.Y = self.model.addVars(self.phi_list, lb=0.0, vtype=VType.CONTINUOUS, name="Y")

        # 第二阶段决策变量
        # Rϕt 变量：根据 use_lp_relaxation 决定是连续变量还是二元变量
        self.R = self.model.addVars(
            [(phi, t) for phi in self.phi_list for t in self.t_list],
            lb=0.0, name="R"
        )

        # Gϕtp 变量：根据 use_lp_relaxation 决定是连续变量还是二元变量
        if self.use_lp_relaxation:
            self.G = self.model.addVars(
                [(phi, t, p) for phi in self.phi_list for t in self.t_list for p in self.p_list],
                lb=0.0, ub=1.0, name="G"
            )
        else:
            self.G = self.model.addVars(
                [(phi, t, p) for phi in self.phi_list for t in self.t_list for p in self.p_list],
                vtype=GRB.BINARY, name="G"
            )

    # ---------------- Objective ----------------
    @timeit_if_debug
    def set_objective(self):
        """
        设置目标函数 (公式 1)
        Stage I: Σ_{ϕ ∈ Φ} Xϕ
        Stage II: Σ_{ϕ ∈ Φ} Σ_{t ∈ T} Σ_{p ∈ P} c_ϕtp G_ϕtp
        """
        long_haul_revenue = gp.quicksum(self.p_hat[phi] * self.X[phi] for phi in self.phi_list)
        ad_hoc_revenue = gp.quicksum(
            self.c_phi_t_p.get((phi, t, p), 0.0) * self.G[(phi, t, p)]
            for phi in self.phi_list
            for t in self.t_list
            for p in self.p_list
        )
        self.model.setObjective(long_haul_revenue + ad_hoc_revenue, GRB.MAXIMIZE)


    # ---------------- Constraints ----------------
    @timeit_if_debug
    def add_constraints(self):
        """添加模型约束。"""
        # Stage I:
        # 1. 网络容量约束 (公式 2)
        self.add_capacity_constr()

        # Stage II:
        # 2. 可用仓位 R 的定义 (公式 3)
        self.add_inventory_constr()

        # 3. 服务水平约束 (公式 4)
        self.add_service_constr()

        # 4. 混合约束 (公式 5)
        self.add_mixing_constr()

    @timeit_if_debug
    def add_inventory_constr(self):
        """
        # 2. 可用仓位计算 (约束3 )
        """
        for phi in self.phi_list:
            for t in self.t_list:
                if 1 <= t <= self.t_d_phi.get(phi, 0):
                    # Rϕt = Yϕ - Σ_{t'=0}^{t-1} [dϕt' - a(Σp p Gϕt'p - p̂_ϕ)]
                    demand_expr = gp.quicksum(
                        self.d_0_phi_t.get((phi, tp), 0.0)
                        - self.a * (
                            gp.quicksum(p * self.G[(phi, tp, p)] for p in self.p_list)
                            - self.p_hat[phi]
                        )
                        for tp in self.t_list if tp < t
                    )
                    self.model.addConstr(
                        self.R[(phi, t)] == self.Y[phi] - demand_expr,
                        name=f"inventory_{phi}_{t}"
                    )
                elif t > self.t_d_phi.get(phi, 0):
                    # t > t_d_phi, Rϕt = 0
                    self.model.addConstr(self.R[(phi, t)] == 0.0, name=f"R_zero_{phi}_{t}")

    @timeit_if_debug
    def add_service_constr(self):
        # 3. 服务水平约束 (公式 4)
        for phi in self.phi_list:
            for t in self.t_list:
                demand = self.d_0_phi_t.get((phi, t), 0.0)
                weighted_price = gp.quicksum(p * self.G[(phi, t, p)] for p in self.p_list)
                lhs = demand - self.a * (weighted_price - self.p_hat[phi])
                self.model.addConstr(lhs <= self.R[(phi, t)], name=f"service_{phi}_{t}")

    @timeit_if_debug
    def add_mixing_constr(self):
        # 4. 混合约束 (公式 5)
        for phi in self.phi_list:
            for t in self.t_list:
                self.model.addConstr(
                    gp.quicksum(self.G[(phi, t, p)] for p in self.p_list) <= 1.0,
                    name=f"mix_{phi}_{t}"
                )

    @timeit_if_debug
    def add_capacity_constr(self):
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

    def get_status(self):
        if (self.model.status == gp.GRB.OPTIMAL):
            return "OPTIMAL"
        elif self.model.status == GRB.INFEASIBLE:
            self.model.computeIIS()
            self.model.write("iis.ilp")
            return "INFEASIBLE"
        elif self.model.status == GRB.UNBOUNDED:
            return  "UNBOUNDED"
        elif self.model.status == GRB.TIME_LIMIT:
            return  "TIME_LIMIT"
        elif self.model.status == GRB.INF_OR_UNBD:
            self.model.write("DM.lp")
            return  "INF_OR_UNBD"
        else:
            return "Other"

    def print_model_info(self):
        """打印模型信息。"""
        print(f"变量数量: {self.model.NumVars}")
        print(f"0-1变量数量: {self.model.NumBinVars}")
        print(f"约束数量: {self.model.NumConstrs}")

    def solve(self, verbose=True):
        """
        求解模型
        """
        self.model.optimize()
        self.solutions = self.get_solution()

    def get_solution(self) -> Dict[str, Any]:
        """获取模型的解。"""
        if self.model.status != GRB.OPTIMAL:
            return {}

        solution = {
            'obj_val': self.model.objVal,
            'X': {phi: self.X[phi].X for phi in self.phi_list},
            'Y': {phi: self.Y[phi].X for phi in self.phi_list},
            'R': {(phi, t): self.R[(phi, t)].X for phi in self.phi_list for t in self.t_list},
            'G': {(phi, t, p): self.G[(phi, t, p)].X for phi in self.phi_list for t in self.t_list for p in self.p_list}
        }

        self.solutions = solution
        return solution

    def write(self):
        self.model.write("DM.lp")


# ================= 使用示例 =================
def main():
    """
    主函数：创建测试数据，构建模型，求解并验证。
    """
    print("=" * 60)
    print("🧪 开始测试 DeterministicModel")
    print("=" * 60)

    # 1. 创建测试参数
    model_params = generate_feasible_test_case(
            num_paths=1,
            num_periods=2,
            num_prices=1,
            uncertainty_dim=1,
            uncertainty_std_ratio=0,
            seed=42  # 固定种子以复现结果
        )
    print("✅ 测试参数已创建。")

    # 2. 构建并求解确定性松弛模型 (LP Relaxation)
    print("\n--- 求解 LP Relaxation 版本 ---")
    det_model_lp = DeterministicModel(model_params, use_lp_relaxation=True)
    det_model_lp.build_model()

    det_model_lp.solve()
    if det_model_lp.get_status() == 'OPTIMAL':
        solution_lp = det_model_lp.get_solution()
        print(f"✅ LP Relaxation 求解成功! 目标值: {solution_lp['obj_val']:.4f}")
    else:
        print("❌ LP Relaxation 求解失败。")

    # 3. (可选) 构建并求解整数规划版本
    print("\n--- 求解 Integer Programming 版本 (可能较慢) ---")
    det_model_ip = DeterministicModel(model_params, use_lp_relaxation=False)
    det_model_ip.build_model()

    det_model_lp.solve()
    if det_model_lp.get_status() == 'OPTIMAL':
        solution_milp = det_model_lp.get_solution()
        print(f"✅ Integer Programming 求解成功! 目标值: {solution_milp['obj_val']:.4f}")

        # 比较两个目标值
    else:
        print("❌ Integer Programming 求解失败。")

    print("\n" + "=" * 60)
    print("🎉 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
