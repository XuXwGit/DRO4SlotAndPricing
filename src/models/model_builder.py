from functools import wraps
import logging
import time
from gurobipy import GRB, Model

from src.config.config import Config

# 👇 定义装饰器在类外，作为模块级函数
def timeit_if_debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            if getattr(Config, 'debug_mode', False):  # 防止 Config 没有 debug_mode
                start = time.perf_counter()
                result = func(*args, **kwargs)
                end = time.perf_counter()
                # 获取类实例的 info 属性（如果有）用于日志标识
                instance = args[0] if len(args) > 0 else None
                info = getattr(instance, 'info', '') if instance else ''
                logging.info(f"[TIME][{info}] {func.__name__} 执行耗时: {end - start:.4f} 秒")
                return result
            else:
                return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"执行 {func.__name__} 时发生错误：{e}")
            raise e
    return wrapper


OPTIMAL = GRB.OPTIMAL
MAXIMIZE = GRB.MAXIMIZE

class VType:
    CONTINUOUS = GRB.CONTINUOUS
    BINARY = GRB.BINARY
    INTEGER = GRB.INTEGER

    def __init__(self, solver = "gurobi") -> None:
        if solver == "gurobi":
            self.CONTINUOUS = GRB.CONTINUOUS
            self.BINARY = GRB.BINARY
            self.INTEGER = GRB.INTEGER
        elif solver == "moesk":
            self.CONTINUOUS = "C"
            self.BINARY = "B"
            self.INTEGER = "I"

class ModelBuilder:
    def __init__(self, info=""):
        self.model = Model(info)
        self.model.setParam('TimeLimit', Config.default_time_limit)
        self.model.setParam('MIPGap', Config.default_mip_gap)
        self.model.setParam('Outputflag', Config.default_output_flag)
        self.info = info
        self.obj_val = 0
        self.solutions = {}
        # self.INF = GRB.INFINITY
        self.INF = 100000000

    @timeit_if_debug
    def create_variables(self):
        """ 创建模型变量 """
        raise NotImplementedError("子类必须实现此方法")

    def set_parameters(self, **kwargs):
        """ 设置模型参数 """
        pass

    @timeit_if_debug
    def set_objective(self):
        """ 设置目标函数 """
        raise NotImplementedError("子类必须实现此方法")

    @timeit_if_debug
    def add_constraints(self):
        """ 添加约束条件 """
        raise NotImplementedError("子类必须实现此方法")

    @timeit_if_debug
    def extract_solution(self):
        """ 提取解 """
        raise NotImplementedError("子类必须实现此方法")

    @timeit_if_debug
    def build_model(self):
        """ 构建模型 """
        try:
            self.set_parameters()

            self.create_variables()

            self.set_objective()

            self.add_constraints()

            self.model.update()
        except Exception as e:
            logging.error(f"构建模型时发生错误：{e}")
            raise e

    def print_model_info(self):
        """
        打印 Gurobi 模型的基本信息：变量数量和约束数量。

        参数:
            model (gurobipy.Model): 已构建的 Gurobi 模型对象。
        """
        num_vars = self.model.NumVars
        num_constrs =self.model.NumConstrs

        print(f"模型信息:")
        print(f"  - 变量数量: {num_vars}")
        print(f"  - 约束数量: {num_constrs}")


    def solve(self):
        """ 求解模型 """
        self.model.optimize()
        self.print_model_status()
        self.extract_solution()


    def get_status(self):
        if (self.model.status == GRB.OPTIMAL):
            return "OPTIMAL"
        elif self.model.status == GRB.INFEASIBLE:
            return "INFEASIBLE"
        elif self.model.status == GRB.UNBOUNDED:
            return  "UNBOUNDED"
        elif self.model.status == GRB.TIME_LIMIT:
            return  "TIME_LIMIT"
        else:
            return "Other"

    def print_model_status(self):
        """ 打印模型状态 """
        if(self.model.status != GRB.OPTIMAL):
            if self.model.status == GRB.INFEASIBLE:
                self.model.computeIIS()
                file_name = self.info
                self.model.write(file_name + '_infeasible'+ '.ilp')
                self.model.write(file_name+ '.lp')
                raise Exception("模型无可行解")
            elif self.model.status == GRB.UNBOUNDED:
                file_name = 'InnerMP'
                self.model.write(file_name+ '.lp')
                logging.warning(f"模型无界，当前目标值：{self.model.ObjVal}") # type: ignore
            elif self.model.SolCount >= 1:
                logging.warning(f"存在可行解，当前目标值：{self.model.ObjVal}") # type: ignore
            elif self.model.status == GRB.TIME_LIMIT:
                logging.warning(f"达到时间限制，当前目标值：{self.model.ObjVal}") # type: ignore
            elif self.model.status == GRB.INF_OR_UNBD:
                self.model.computeIIS()
                file_name = self.info
                self.model.write(file_name + '_infeas_or_Unbound'+ '.ilp')
                self.model.write(file_name+ '.lp')
                logging.warning(f"模型无界或不可行，当前目标值：{self.model.ObjVal}") # type: ignore
            else:
                logging.warning(f"其他状态码：{self.model.status}") # type: ignore
