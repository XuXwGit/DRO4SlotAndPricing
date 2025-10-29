from functools import wraps
import logging
import time
from gurobipy import GRB, Model
import numpy as np

from src.config.config import Config

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
    模型参数设置部分基类完成
    """
    def set_model_params(self, model_params):
        """
        导入模型参数
        @params: model_params: dict 模型参数字典
        """
        pass


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

    """
    =========================================
    以下成员函数需要子类实现
    """
    def update(self):
        """ 更新模型 """
        if self.solver == 'gurobi':
            self.model.update()
        else:
            pass

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

    def get_status(self):
        raise NotImplementedError("子类必须实现此方法")

    def print_model_status(self):
        """ 打印模型状态 """
        raise NotImplementedError("子类必须实现此方法")

    def print_model_info(self):
        """
        打印 模型的基本信息：变量数量和约束数量。
        """
        raise NotImplementedError("子类必须实现此方法")

    @timeit_if_debug
    def solve(self):
        """
        求解模型
        """
        self.model.optimize()
        self.print_model_status()
        self.extract_solution()

    def write(self):
        """ 将模型写入文件 """
        pass