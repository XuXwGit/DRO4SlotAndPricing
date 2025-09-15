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

class DefaultSetting:
    """
    默认设置类
    对应Java类: multi.DefaultSetting
    
    定义系统运行所需的默认参数和配置
    """
    
    ##################################
    # 数值实验测试 - 对应Java中的Numerical Experiment Test部分
    ##################################
    # 船舶容量范围 - 对应Java: public static String VesselCapacityRange = "I";
    VESSEL_CAPACITY_RANGE = "I" 
    # 船队类型：Homo/Hetero - 对应Java: public static String FleetType = "Hetero";
    FLEET_TYPE = "Homo"        
    
    ##################################
    # 集装箱设置 - 对应Java中的container settings
    ##################################
    # 是否允许折叠箱 - 对应Java: public static boolean AllowFoldableContainer = true;
    ALLOW_FOLDABLE_CONTAINER = True  
    # 空箱调度方式：是否重定向 - 对应Java: public static boolean IsEmptyReposition = false;
    IS_EMPTY_REPOSITION = False      
    
    ##################################
    # 策略设置 - 对应Java中的Strategy DefaultSetting
    ##################################
    # 路径减少百分比 - 对应Java: public static double reducePathPercentage = 0;
    REDUCE_PATH_PERCENTAGE = 0       
    # 最大重箱路径数 - 对应Java: public static int MaxLadenPathsNum = 5;
    MAX_LADEN_PATHS_NUM = 5         
    # 最大空箱路径数 - 对应Java: public static int MaxEmptyPathsNum = 5;
    MAX_EMPTY_PATHS_NUM = 5         
    # 是否使用帕累托最优切割 - 对应Java: public static boolean UseParetoOptimalCut = true;
    USE_PARETO_OPTIMAL_CUT = True   
    # 是否使用局部搜索 - 对应Java: public static boolean UseLocalSearch = true;
    USE_LOCAL_SEARCH = True         
    
    ##################################
    # 默认数据参数设置 - 对应Java中的Default Data Parameter Setting
    ##################################
    # 默认单位租赁成本 - 对应Java: public static int DefaultUnitRentalCost = 50;
    DEFAULT_UNIT_RENTAL_COST = 50           
    # 默认重箱滞期成本 - 对应Java: public static int DefaultLadenDemurrageCost = 175;
    DEFAULT_LADEN_DEMURRAGE_COST = 175      
    # 默认空箱滞期成本 - 对应Java: public static int DefaultEmptyDemurrageCost = 100;
    DEFAULT_EMPTY_DEMURRAGE_COST = 100      
    # 默认单位装载成本 - 对应Java: public static int DefaultUnitLoadingCost = 20;
    DEFAULT_UNIT_LOADING_COST = 20          
    # 默认单位卸载成本 - 对应Java: public static int DefaultUnitDischargeCost = 20;
    DEFAULT_UNIT_DISCHARGE_COST = 20        
    # 默认单位转运成本 - 对应Java: public static int DefaultUnitTransshipmentCost = 30;
    DEFAULT_UNIT_TRANSSHIPMENT_COST = 30    
    # 默认计划期 - 对应Java: public static int DefaultTimeHorizon = 60;
    DEFAULT_TIME_HORIZON = 60
    # 默认周转时间 - 对应Java: public static int DefaultTurnOverTime = 14;
    DEFAULT_TURN_OVER_TIME = 14             
    # 默认折叠箱比例 - 对应Java: public static double DefaultFoldContainerPercent = 0.15;
    DEFAULT_FOLD_CONTAINER_PERCENT = 0.15   
    # 默认折叠空箱成本偏差 - 对应Java: public static double DefaultFoldEmptyCostBias = 15;
    DEFAULT_FOLD_EMPTY_COST_BIAS = 15       
    
    ##################################
    # 调试设置 - 对应Java中的debug settings
    ##################################
    # 是否启用调试 - 对应Java: public static boolean DebugEnable = false;
    DEBUG_ENABLE = False            
    # 是否在生成参数时显示设置信息 - 对应Java: public static boolean GenerateParamEnable = false;
    GENERATE_PARAM_ENABLE = False   
    # 是否在子问题中显示设置信息 - 对应Java: public static boolean SubEnable = true;
    SUB_ENABLE = True              
    # 是否在对偶问题中显示设置信息 - 对应Java: public static boolean DualEnable = false;
    DUAL_ENABLE = False            
    # 是否在对偶子问题中显示设置信息 - 对应Java: public static boolean DualSubEnable = true;
    DUAL_SUB_ENABLE = True         
    # 是否在主问题中显示设置信息 - 对应Java: public static boolean MasterEnable = false;
    MASTER_ENABLE = False          
    
    ##################################
    # 输入数据设置 - 对应Java中的Input Data Settings
    ##################################
    # 请求包含范围 - 对应Java: public static double RequestIncludeRange = 0;
    REQUEST_INCLUDE_RANGE = 0               
    # 是否允许同区域转运 - 对应Java: public static boolean WhetherAllowSameRegionTrans = true;
    WHETHER_ALLOW_SAME_REGION_TRANS = True  
    # 是否切割超成本路径 - 对应Java: public static boolean WhetherCuttingOverCostPaths = true;
    WHETHER_CUTTING_OVER_COST_PATHS = False 
    
    ##################################
    # 随机设置 - 对应Java中的Random Setting
    ##################################
    # 分布类型：Log-Normal/Uniform/Normal - 对应Java: public static String distributionType = "Uniform";
    DISTRIBUTION_TYPE = "Uniform"           
    # 随机数生成器 - 对应Java: public static Random random;
    random = None
    # 随机种子 - 对应Java: public static int randomSeed = 0;
    RANDOM_SEED = 0                         
    # 是否生成样本 - 对应Java: public static boolean WhetherGenerateSamples = true;
    WHETHER_GENERATE_SAMPLES = True         
    # 是否计算平均性能 - 对应Java: public static boolean WhetherCalculateMeanPerformance = false;
    WHETHER_CALCULATE_MEAN_PERFORMANCE = True  
    # 是否写入样本测试 - 对应Java: public static boolean WhetherWriteSampleTests = true;
    WHETHER_WRITE_SAMPLE_TESTS = True      
    # 是否加载样本测试 - 对应Java: public static boolean WhetherLoadSampleTests = false;
    WHETHER_LOAD_SAMPLE_TESTS = False       
    # 样本场景数量 - 对应Java: public static int numSampleScenes = 10;
    NUM_SAMPLE_SCENES = 10                  
    # 对数正态分布sigma因子 - 对应Java: public static double log_normal_sigma_factor = 1.0;
    LOG_NORMAL_SIGMA_FACTOR = 1.0           
    # 预算系数 - 对应Java: public static double budgetCoefficient = 1.0;
    BUDGET_COEFFICIENT = 1.0                
    # 默认不确定度 - 对应Java: public static double defaultUncertainDegree = 0.15;
    DEFAULT_UNCERTAIN_DEGREE = 0.15         
    # 惩罚系数 - 对应Java: public static double penaltyCoefficient = 1.0;
    PENALTY_COEFFICIENT = 1.0               
    # 初始空箱数量 - 对应Java: public static int initialEmptyContainers = 28;
    INITIAL_EMPTY_CONTAINERS = 28           
    # 鲁棒性 - 对应Java: public static int robustness = 1;
    ROBUSTNESS = 100
    # 重箱滞期免费时间 - 对应Java: public static int ladenStayFreeTime = 7;
    LADEN_STAY_FREE_TIME = 7
    # 空箱滞期免费时间 - 对应Java: public static int emptyStayFreeTime = 7;
    EMPTY_STAY_FREE_TIME = 7

    ##################################
    # 日志设置 - 对应Java中的logging settings
    ##################################
    # 是否写入文件日志 - 对应Java: public static boolean WhetherWriteFileLog = false;
    WHETHER_WRITE_FILE_LOG = False          
    # 是否打印文件日志 - 对应Java: public static boolean WhetherPrintFileLog = false;
    WHETHER_PRINT_FILE_LOG = False          
    # 是否打印数据状态 - 对应Java: public static boolean WhetherPrintDataStatus = false;
    WHETHER_PRINT_DATA_STATUS = False       
    # 是否打印船舶决策 - 对应Java: public static boolean WhetherPrintVesselDecision = false;
    WHETHER_PRINT_VESSEL_DECISION = False   
    # 是否打印请求决策 - 对应Java: public static boolean WhetherPrintRequestDecision = false;
    WHETHER_PRINT_REQUEST_DECISION = False  
    # 是否打印迭代信息 - 对应Java: public static boolean WhetherPrintIteration = true;
    WHETHER_PRINT_ITERATION = True          
    # 是否打印求解时间 - 对应Java: public static boolean WhetherPrintSolveTime = false;
    WHETHER_PRINT_SOLVE_TIME = False        
    # 是否打印处理过程 - 对应Java: public static boolean WhetherPrintProcess = true;
    WHETHER_PRINT_PROCESS = True            
    
    ##################################
    # CPLEX求解器设置 - 对应Java中的Cplex Solver Settings
    ##################################
    # 是否导出模型 - 对应Java: public static boolean WhetherExportModel = false;
    WHETHER_EXPORT_MODEL = False            
    # 是否关闭输出日志 - 对应Java: public static boolean WhetherCloseOutputLog = true;
    WHETHER_CLOSE_OUTPUT_LOG = True         
    # MIP求解间隙限制 - 对应Java: public static double MIPGapLimit = 1e-3;
    MIP_GAP_LIMIT = 1e-3                    
    # MIP求解时间限制（秒） - 对应Java: public static double MIPTimeLimit = 36000;
    MIP_TIME_LIMIT = 36000                  
    # 最大线程数 - 对应Java: public static int MaxThreads = Runtime.getRuntime().availableProcessors();
    MAX_THREADS = os.cpu_count()            
    # 最大工作内存 - 对应Java: public static long MaxWorkMem = (Runtime.getRuntime().maxMemory() >> 20);
    MAX_WORK_MEM = sys.maxsize >> 20        
    
    ##################################
    # 算法设置 - 对应Java中的Algo DefaultSetting
    ##################################
    # 算法类型 - 对应Java: public static String algorithm = "bd";
    DEFAULT_ALGORITHM = "bd"
    # 最大迭代次数 - 对应Java: public static int maxIterationNum = 100;
    MAX_ITERATION_NUM = 100                 
    # 最大迭代时间 - 对应Java: public static int maxIterationTime = 3600;
    MAX_ITERATION_TIME = 3600               
    # 边界间隙限制 - 对应Java: public static double boundGapLimit = 1.0;
    BOUND_GAP_LIMIT = 1.0                   
    # 是否设置初始解 - 对应Java: public static boolean WhetherSetInitialSolution = false;
    WHETHER_SET_INITIAL_SOLUTION = False    
    # 是否添加初始化场景 - 对应Java: public static boolean WhetherAddInitializeSce = false;
    WHETHER_ADD_INITIALIZE_SCE = False      
    # 是否使用CCG-PAP-SP - 对应Java: public static boolean CCG_PAP_Use_Sp = true;
    CCG_PAP_USE_SP = True                   
    # 是否使用历史解 - 对应Java: public static boolean UseHistorySolution = false;
    USE_HISTORY_SOLUTION = False            
    
    ##################################
    # Python编程设置 - 对应Java中的Java Programming DefaultSetting
    ##################################
    # 是否使用多线程 - 对应Java: public static boolean WhetherUseMultiThreads = false;
    WHETHER_USE_MULTI_THREADS = True        
    # 进度条宽度 - 对应Java: public static int ProgressBarWidth = 50;
    PROGRESS_BAR_WIDTH = 50                 
    
    ##################################
    # 路径设置 - 对应Java中的Root path
    ##################################
    # 根路径 - 对应Java: public static String RootPath = System.getProperty("user.dir") + "/";
    ROOT_PATH = os.getcwd() + "/"           
    # 数据路径 - 对应Java: public static String DataPath = "data/";
    DATA_PATH = "data/"                     
    # 案例路径 - 对应Java: public static String CasePath = "1/";
    CASE_PATH = "1/"                        
    # 模型导出路径 - 对应Java: public static String ExportModelPath = "model/";
    EXPORT_MODEL_PATH = "model/"
    # 算法日志路径 - 对应Java: public static String AlgoLogPath = "log/";
    ALGO_LOG_PATH = "log/"                  
    # 解决方案路径 - 对应Java: public static String SolutionPath = "solution/";
    SOLUTION_PATH = "solution/"             
    # 测试结果路径 - 对应Java: public static String TestResultPath = "result/";
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
        对应Java方法: public static void drawProgressBar(int progress)
        
        Args:
            progress: 进度百分比(0-100)
        """
        # 计算已完成的进度条数量
        # 对应Java: int completedBars = progress * ProgressBarWidth / 100;
        completed_bars = progress * DefaultSetting.PROGRESS_BAR_WIDTH // 100
        
        # 构建进度条字符串
        # 对应Java: StringBuilder progressBar = new StringBuilder();
        # 对应Java: progressBar.append("\r[");
        progress_bar = ["\r["]
        
        # 填充进度条
        # 对应Java: for (int i = 0; i < ProgressBarWidth; i++) {
        for i in range(DefaultSetting.PROGRESS_BAR_WIDTH):
            # 对应Java: if (i < completedBars) { progressBar.append("="); }
            if i < completed_bars:
                progress_bar.append("=")
            # 对应Java: else if (i == completedBars) { progressBar.append(">"); }
            elif i == completed_bars:
                progress_bar.append(">")
            # 对应Java: else { progressBar.append("   "); }
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
        对应Java方法: public static void updateSetting(String attr, Object value)
        """
        # 模型参数
        if attr == "time_window":
            DefaultSetting.DEFAULT_TIME_HORIZON = value
        elif attr == "turn_over_time":
            DefaultSetting.DEFAULT_TURN_OVER_TIME = value
        elif attr == "robustness":
            DefaultSetting.ROBUSTNESS = value
        elif attr == "demand_fluctuation":
            DefaultSetting.DEFAULT_UNCERTAIN_DEGREE = value
        elif attr == 'empty_rent_cost':
            DefaultSetting.DEFAULT_UNIT_RENTAL_COST = value
        elif attr == 'penalty_coeff':
            DefaultSetting.PENALTY_COEFFICIENT = value
        elif attr == 'port_load_cost':
            DefaultSetting.LOADING_COST = value
        elif attr == 'port_unload_cost':
            DefaultSetting.DEFAULT_UNIT_DISCHARGE_COST = value
        elif attr == 'port_transship_cost':
            DefaultSetting.DEFAULT_UNIT_TRANSSHIPMENT_COST = value
        elif attr == 'laden_stay_cost':
            DefaultSetting.DEFAULT_LADEN_DEMURRAGE_COST = value
        elif attr == 'empty_stay_cost':
            DefaultSetting.DEFAULT_EMPTY_DEMURRAGE_COST = value
        elif attr == 'laden_stay_free_time':
            DefaultSetting.LADEN_STAY_FREE_TIME = value
        elif attr == 'empty_stay_free_time':
            DefaultSetting.EMPTY_STAY_FREE_TIME = value
            
        # 算法参数
        elif attr == 'algorithm':
            DefaultSetting.DEFAULT_ALGORITHM = value
        elif attr == 'max_iter':
            DefaultSetting.MAX_ITERATION_NUM = value
        elif attr == 'max_time':
            DefaultSetting.MAX_ITERATION_TIME = value
        elif attr == 'mip_gap':
            DefaultSetting.MIP_GAP_LIMIT = value

    @staticmethod
    def update_setting_from_args(args):
        for key, value in vars(args).items():
            # 根据 key 和 value 更新 DefaultSetting
            DefaultSetting.update_setting(attr=key, value=value)


    @staticmethod
    def update_setting_from_dict(setting_dict: Dict[str, Any]) -> None:
        """
        从字典更新设置
        对应Java方法: public static void updateSettingFromDict(Map<String, Object> settingDict)
        """
        for key, value in setting_dict.items():
            try:
                DefaultSetting.update_setting(key, value)
            except Exception as e:
                logger.error(f"更新设置失败: {e}")

    @staticmethod
    def print_settings() -> None:
        """
        打印基本设置信息
        对应Java方法: public static void printSettings()
        """
        # 对应Java: log.info("======================"+ "Settings" + "======================");
        logger.info("======================Settings======================")
        
        # 对应Java: log.info("FleetType = " + FleetType);
        logger.info(f"FleetType = {DefaultSetting.FLEET_TYPE}")
        
        # 对应Java: log.info("VesselType Set = " + VesselCapacityRange);
        logger.info(f"VesselType Set = {DefaultSetting.VESSEL_CAPACITY_RANGE}")
        
        # 对应Java: log.info("Random Distribution = " + distributionType);
        logger.info(f"Random Distribution = {DefaultSetting.DISTRIBUTION_TYPE}")
        
        # 对应Java: log.info("MIPGapLimit = " + MIPGapLimit);
        logger.info(f"MIPGapLimit = {DefaultSetting.MIP_GAP_LIMIT}")
        
        # 对应Java: log.info("MIPTimeLimit = " + MIPTimeLimit + "s");
        logger.info(f"MIPTimeLimit = {DefaultSetting.MIP_TIME_LIMIT}s")
        
        # 对应Java: log.info("MaxThreads = " + MaxThreads);
        logger.info(f"MaxThreads = {DefaultSetting.MAX_THREADS}")
        
        # 对应Java: log.info("MaxWorkMem = " + MaxWorkMem +"M");
        logger.info(f"MaxWorkMem = {DefaultSetting.MAX_WORK_MEM}M")
        
        # 对应Java: log.info("NumSampleScenes = " + numSampleScenes);
        logger.info(f"NumSampleScenes = {DefaultSetting.NUM_SAMPLE_SCENES}")
        
        # 对应Java: log.info("maxIterationNum = " + maxIterationNum);
        logger.info(f"maxIterationNum = {DefaultSetting.MAX_ITERATION_NUM}")
        
        # 对应Java: log.info("maxIterationTime = " + maxIterationTime +"s");
        logger.info(f"maxIterationTime = {DefaultSetting.MAX_ITERATION_TIME}s")
        
        # 对应Java: log.info("boundGapLimit = " + boundGapLimit);
        logger.info(f"boundGapLimit = {DefaultSetting.BOUND_GAP_LIMIT}")
        
        # 对应Java: log.info("RandomSeed = "+randomSeed);
        logger.info(f"RandomSeed = {DefaultSetting.RANDOM_SEED}")
        
        # 对应Java: log.info("WhetherLoadHistorySolution = " + UseHistorySolution);
        logger.info(f"WhetherLoadHistorySolution = {DefaultSetting.USE_HISTORY_SOLUTION}")
        
        # 对应Java: log.info("WhetherAddInitializeSce = " + WhetherAddInitializeSce);
        logger.info(f"WhetherAddInitializeSce = {DefaultSetting.WHETHER_ADD_INITIALIZE_SCE}")

            
    @staticmethod
    def init_random(seed: int = 42) -> None:
        """
        初始化随机数生成器
        对应Java: public static void initRandom(Integer seed)
        
        Args:
            seed: 随机数种子，如果为None则使用默认种子
        """
        # 如果提供了种子，则使用该种子，否则使用默认种子
        # 对应Java: if (seed != null) { randomSeed = seed; }
        if seed is not None:
            DefaultSetting.RANDOM_SEED = seed
        
        # 初始化随机数生成器
        # 对应Java: random = new Random(randomSeed);
        DefaultSetting.random = random.Random(DefaultSetting.RANDOM_SEED)
        
        # 记录随机数种子信息
        # 对应Java: log.info("Random seed: " + randomSeed);
        logger.info(f"Random seed: {DefaultSetting.RANDOM_SEED}") 