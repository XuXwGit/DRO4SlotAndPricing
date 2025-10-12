"""
@Author: XuXw
@Description: 默认设置类
@DateTime: 2024/12/4 21:54
"""
import argparse
import logging
import os
import random
import sys
from typing import List, Dict, Any, TextIO


# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

class Config:
    """
    默认设置类
    
    定义系统运行所需的默认参数和配置
    """
    debug_mode = True

    """ 设置求解器参数 """
    default_solver = 'gurobi'                     # 默认求解器
    default_mip_gap = 0.01                        # 默认求解器Gap限制
    default_time_limit = 3600                     # 默认求解时间限制为1小时
    default_output_flag = True                 # 默认输出求解结果

    ##################################
    # 数值实验测试
    ##################################
    # 船舶容量范围
    VESSEL_CAPACITY_RANGE = "I" 
    # 船队类型：Homo/Hetero
    FLEET_TYPE = "Homo" 

    
    ##################################
    # 集装箱设置
    ##################################
    # 是否允许折叠箱
    # 空箱调度方式：是否重定向
    IS_EMPTY_REPOSITION = False      
    
    ##################################
    # 策略设置
    ##################################
    # 路径减少百分比
    REDUCE_PATH_PERCENTAGE = 0       
    # 最大重箱路径数
    MAX_LADEN_PATHS_NUM = 5         
    # 最大空箱路径数
    MAX_EMPTY_PATHS_NUM = 5         
    # 是否使用帕累托最优切割
    USE_PARETO_OPTIMAL_CUT = True   
    # 是否使用局部搜
    USE_LOCAL_SEARCH = True         
    
    ##################################
    # 默认数据参数设置
    ##################################
    # 默认单位租赁成本
    DEFAULT_UNIT_RENTAL_COST = 50           
    # 默认重箱滞期成本
    DEFAULT_LADEN_DEMURRAGE_COST = 175      
    # 默认空箱滞期成本
    DEFAULT_EMPTY_DEMURRAGE_COST = 100      
    # 默认单位装载成本
    DEFAULT_UNIT_LOADING_COST = 20          
    # 默认单位卸载成本
    DEFAULT_UNIT_DISCHARGE_COST = 20        
    # 默认单位转运成本
    DEFAULT_UNIT_TRANSSHIPMENT_COST = 30    
    # 默认计划期
    DEFAULT_TIME_HORIZON = 60
    # 默认周转时间
    DEFAULT_TURN_OVER_TIME = 14             
    # 默认折叠箱比例
    DEFAULT_FOLD_CONTAINER_PERCENT = 0.15   
    # 默认折叠空箱成本偏差
    DEFAULT_FOLD_EMPTY_COST_BIAS = 15       
    
    ##################################
    # 调试设置
    ##################################
    # 是否启用调试 
    DEBUG_ENABLE = False            
    # 是否在生成参数时显示设置信息 
    GENERATE_PARAM_ENABLE = False   
    # 是否在子问题中显示设置信息 
    SUB_ENABLE = True              
    # 是否在对偶问题中显示设置信息 
    DUAL_ENABLE = False            
    # 是否在对偶子问题中显示设置信息 
    DUAL_SUB_ENABLE = True         
    # 是否在主问题中显示设置信息 
    MASTER_ENABLE = False          
    
    ##################################
    # 输入数据设置 
    ##################################
    # 请求包含范围 
    REQUEST_INCLUDE_RANGE = 0               
    # 是否允许同区域转运 
    WHETHER_ALLOW_SAME_REGION_TRANS = True  
    # 是否切割超成本路径 
    WHETHER_CUTTING_OVER_COST_PATHS = False 
    
    ##################################
    # 随机设置 
    ##################################
    # 分布类型：Log-Normal/Uniform/Normal 
    DISTRIBUTION_TYPE = "Uniform"           
    # 随机数生成器 
    random = None
    # 随机种子 
    RANDOM_SEED = 0                         
    # 是否生成样本 
    WHETHER_GENERATE_SAMPLES = True         
    # 是否计算平均性能 
    WHETHER_CALCULATE_MEAN_PERFORMANCE = True  
    # 是否写入样本测试 
    WHETHER_WRITE_SAMPLE_TESTS = True      
    # 是否加载样本测试 
    WHETHER_LOAD_SAMPLE_TESTS = False       
    # 样本场景数量 
    NUM_SAMPLE_SCENES = 10                  
    # 对数正态分布sigma因子 
    LOG_NORMAL_SIGMA_FACTOR = 1.0           
    # 预算系数 
    BUDGET_COEFFICIENT = 1.0                
    # 默认不确定度 
    DEFAULT_UNCERTAIN_DEGREE = 0.15         
    # 惩罚系数 
    PENALTY_COEFFICIENT = 1.0               
    # 初始空箱数量 
    INITIAL_EMPTY_CONTAINERS = 28           
    # 鲁棒性 
    ROBUSTNESS = 100
    # 重箱滞期免费时间 
    LADEN_STAY_FREE_TIME = 7
    # 空箱滞期免费时间 
    EMPTY_STAY_FREE_TIME = 7

    ##################################
    # 日志设置 
    ##################################
    # 是否写入文件日志 
    WHETHER_WRITE_FILE_LOG = False          
    # 是否打印文件日志 
    WHETHER_PRINT_FILE_LOG = False          
    # 是否打印数据状态 
    WHETHER_PRINT_DATA_STATUS = False       
    # 是否打印船舶决策 
    WHETHER_PRINT_VESSEL_DECISION = False   
    # 是否打印请求决策 
    WHETHER_PRINT_REQUEST_DECISION = False  
    # 是否打印迭代信息 
    WHETHER_PRINT_ITERATION = True          
    # 是否打印求解时间 
    WHETHER_PRINT_SOLVE_TIME = False        
    # 是否打印处理过程 
    WHETHER_PRINT_PROCESS = True            
    
    ##################################
    # CPLEX求解器设置 
    ##################################
    # 是否导出模型 
    WHETHER_EXPORT_MODEL = False            
    # 是否关闭输出日志 
    WHETHER_CLOSE_OUTPUT_LOG = True         
    # MIP求解间隙限制 
    MIP_GAP_LIMIT = 1e-3                    
    # MIP求解时间限制（秒） 
    MIP_TIME_LIMIT = 36000                  
    # 最大线程数 
    MAX_THREADS = os.cpu_count()            
    # 最大工作内存 
    MAX_WORK_MEM = sys.maxsize >> 20        
    
    ##################################
    # 算法设置 
    ##################################
    # 算法类型 
    DEFAULT_ALGORITHM = "bd"
    # 最大迭代次数 
    MAX_ITERATION_NUM = 100                 
    # 最大迭代时间 
    MAX_ITERATION_TIME = 3600               
    # 边界间隙限制 
    BOUND_GAP_LIMIT = 1.0                   
    # 是否设置初始解 
    WHETHER_SET_INITIAL_SOLUTION = False    
    # 是否添加初始化场景 
    WHETHER_ADD_INITIALIZE_SCE = False      
    # 是否使用CCG-PAP-SP 
    CCG_PAP_USE_SP = True                   
    # 是否使用历史解 
    USE_HISTORY_SOLUTION = False            
    
    ##################################
    # Python编程设置 
    ##################################
    # 是否使用多线程 
    WHETHER_USE_MULTI_THREADS = True        
    # 进度条宽度 
    PROGRESS_BAR_WIDTH = 50                 
    
    ##################################
    # 路径设置 
    ##################################
    # 根路径 
    ROOT_PATH = os.getcwd() + "/"           
    # 数据路径 
    DATA_PATH = "data/"                     
    # 案例路径 
    CASE_PATH = "1/"                        
    # 模型导出路径 
    EXPORT_MODEL_PATH = "model/"
    # 算法日志路径 
    ALGO_LOG_PATH = "log/"                  
    # 解决方案路径 
    SOLUTION_PATH = "solution/"             
    # 测试结果路径 
    TEST_RESULT_PATH = "result/"            
    
    # ================== 数据文件名常量 ==================
    PORTS_FILENAME = "Ports.txt"
    ROUTES_FILENAME = "ShippingRoutes.txt"
    NODES_FILENAME = "Nodes.txt"
    TRAVELING_ARCS_FILENAME = "TravelingArcs.txt"
    TRANSSHIP_ARCS_FILENAME = "TransshipArcs.txt"
    VESSEL_PATHS_FILENAME = "VesselPaths.txt"
    LADEN_PATHS_FILENAME = "LadenPaths.txt"
    EMPTY_PATHS_FILENAME = "EmptyPaths.txt"
    REQUESTS_FILENAME = "Requests.txt"
    VESSELS_FILENAME = "Vessels.txt"
    DEMAND_RANGE_FILENAME = "DemandRange.txt"
    PATHS_FILENAME = "Paths.txt"
    HISTORY_SOLUTION_FILENAME = "HistorySolution.txt"
    SAMPLE_SCENES_FILENAME = "SampleScenes.txt"
    
    # ================== 文件名称-数据库表名映射 ==================
    FILE_TABLE_MAP = {
        PORTS_FILENAME: "ports",
        ROUTES_FILENAME: "routes",
        VESSELS_FILENAME: "ships",
        NODES_FILENAME: "nodes",
        TRAVELING_ARCS_FILENAME: "traveling_arcs",
        TRANSSHIP_ARCS_FILENAME: "transship_arcs",
        VESSEL_PATHS_FILENAME: "vessel_paths",
        LADEN_PATHS_FILENAME: "laden_paths",
        EMPTY_PATHS_FILENAME: "empty_paths",
        REQUESTS_FILENAME: "requests",
        PATHS_FILENAME: "paths",
        HISTORY_SOLUTION_FILENAME: "history_solution",
        DEMAND_RANGE_FILENAME: "demand_range",
        SAMPLE_SCENES_FILENAME: "sample_scenes",
    }



    @staticmethod
    def draw_progress_bar(progress: int) -> None:
        """
        在控制台打印带百分比的进度条
        
        
        Args:
            progress: 进度百分比(0-100)
        """
        # 计算已完成的进度条数量
        
        completed_bars = progress * Config.PROGRESS_BAR_WIDTH // 100
        
        # 构建进度条字符串
        
        
        progress_bar = ["\r["]
        
        # 填充进度条
        
        for i in range(Config.PROGRESS_BAR_WIDTH):
            
            if i < completed_bars:
                progress_bar.append("=")
            
            elif i == completed_bars:
                progress_bar.append(">")
            
            else:
                progress_bar.append(" ")
        
        # 添加百分比
        progress_bar.append(f"] {progress}%\r")
        
        # 输出进度条
        print("".join(progress_bar), end="", flush=True)
    

    @staticmethod
    def update_setting(attr: str, value: Any) -> None:
        """
        更新设置
        
        """
        # 模型参数
        if attr == "time_window":
            Config.DEFAULT_TIME_HORIZON = value
        elif attr == "turn_over_time":
            Config.DEFAULT_TURN_OVER_TIME = value
        elif attr == "robustness":
            Config.ROBUSTNESS = value
        elif attr == "demand_fluctuation":
            Config.DEFAULT_UNCERTAIN_DEGREE = value
        elif attr == 'empty_rent_cost':
            Config.DEFAULT_UNIT_RENTAL_COST = value
        elif attr == 'penalty_coeff':
            Config.PENALTY_COEFFICIENT = value
        elif attr == 'port_load_cost':
            Config.LOADING_COST = value
        elif attr == 'port_unload_cost':
            Config.DEFAULT_UNIT_DISCHARGE_COST = value
        elif attr == 'port_transship_cost':
            Config.DEFAULT_UNIT_TRANSSHIPMENT_COST = value
        elif attr == 'laden_stay_cost':
            Config.DEFAULT_LADEN_DEMURRAGE_COST = value
        elif attr == 'empty_stay_cost':
            Config.DEFAULT_EMPTY_DEMURRAGE_COST = value
        elif attr == 'laden_stay_free_time':
            Config.LADEN_STAY_FREE_TIME = value
        elif attr == 'empty_stay_free_time':
            Config.EMPTY_STAY_FREE_TIME = value
            
        # 算法参数
        elif attr == 'algorithm':
            Config.DEFAULT_ALGORITHM = value
        elif attr == 'max_iter':
            Config.MAX_ITERATION_NUM = value
        elif attr == 'max_time':
            Config.MAX_ITERATION_TIME = value
        elif attr == 'mip_gap':
            Config.MIP_GAP_LIMIT = value

    @staticmethod
    def update_setting_from_args(args):
        for key, value in vars(args).items():
            # 根据 key 和 value 更新 DefaultSetting
            Config.update_setting(attr=key, value=value)


    @staticmethod
    def update_setting_from_dict(setting_dict: Dict[str, Any]) -> None:
        """
        从字典更新设置
        
        """
        for key, value in setting_dict.items():
            try:
                Config.update_setting(key, value)
            except Exception as e:
                logger.error(f"更新设置失败: {e}")

    @staticmethod
    def print_settings() -> None:
        """
        打印基本设置信息
        """
        
        logger.info("======================Settings======================")
        logger.info(f"FleetType = {Config.FLEET_TYPE}")
        logger.info(f"VesselType Set = {Config.VESSEL_CAPACITY_RANGE}")
        logger.info(f"Random Distribution = {Config.DISTRIBUTION_TYPE}")
        logger.info(f"MIPGapLimit = {Config.MIP_GAP_LIMIT}")
        logger.info(f"MIPTimeLimit = {Config.MIP_TIME_LIMIT}s")
        logger.info(f"MaxThreads = {Config.MAX_THREADS}")
        logger.info(f"MaxWorkMem = {Config.MAX_WORK_MEM}M")
        logger.info(f"NumSampleScenes = {Config.NUM_SAMPLE_SCENES}")
        logger.info(f"maxIterationNum = {Config.MAX_ITERATION_NUM}")
        logger.info(f"maxIterationTime = {Config.MAX_ITERATION_TIME}s")
        logger.info(f"boundGapLimit = {Config.BOUND_GAP_LIMIT}")
        logger.info(f"RandomSeed = {Config.RANDOM_SEED}")
        logger.info(f"WhetherLoadHistorySolution = {Config.USE_HISTORY_SOLUTION}")
        logger.info(f"WhetherAddInitializeSce = {Config.WHETHER_ADD_INITIALIZE_SCE}")

            
    @staticmethod
    def init_random(seed: int = 42) -> None:
        """
        初始化随机数生成器
        
        
        Args:
            seed: 随机数种子，如果为None则使用默认种子
        """
        # 如果提供了种子，则使用该种子，否则使用默认种子
        
        if seed is not None:
            Config.RANDOM_SEED = seed
        
        # 初始化随机数生成器
        
        Config.random = random.Random(Config.RANDOM_SEED)
        
        # 记录随机数种子信息
        
        logger.info(f"Random seed: {Config.RANDOM_SEED}") 