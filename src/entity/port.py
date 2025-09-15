"""
@Author: XuXw
@Description: Todo
@DateTime: 2024/12/4 21:54
"""


class Port:
    """
    港口类
    
    存储港口相关的属性和信息
    
    属性:
        id: 港口ID
        port: 港口名称
        region: 所属区域
        whether_trans: 是否为转运港
        group: 港口分组
        turn_over_time: 周转时间
        laden_demurrage_cost: 重箱滞期成本
        empty_demurrage_cost: 空箱滞期成本
        loading_cost: 装货成本
        discharge_cost: 卸货成本
        transshipment_cost: 转运成本
        rental_cost: 租赁成本
    """
    
    def __init__(self,
                 id: int = 0,
                 port: str = "",
                 region: str = "",
                 whether_trans: int = 0,
                 group: int = 0,
                 turn_over_time: int = 0,
                 laden_demurrage_cost: float = 0.0,
                 empty_demurrage_cost: float = 0.0,
                 loading_cost: float = 0.0,
                 discharge_cost: float = 0.0,
                 transshipment_cost: float = 0.0,
                 rental_cost: float = 0.0):
        self._id: int = id  # 对应Java: private int id
        self._port: str = port  # 对应Java: private String port
        self._region: str = region  # 对应Java: private String region
        self._whether_trans: int = whether_trans  # 对应Java: private int whetherTrans
        self._group: int = group  # 对应Java: private int group
        self._turn_over_time: int = turn_over_time  # 对应Java: private int turnOverTime
        
        # 成本属性
        self._laden_demurrage_cost: float = laden_demurrage_cost  # 对应Java: private double ladenDemurrangeCost
        self._empty_demurrage_cost: float = empty_demurrage_cost  # 对应Java: private double emptyDemurrangeCost
        self._loading_cost: float = loading_cost  # 对应Java: private double loadingCost
        self._discharge_cost: float = discharge_cost  # 对应Java: private double dischargeCost
        self._transshipment_cost: float = transshipment_cost  # 对应Java: private double transshipmentCost
        self._rental_cost: float = rental_cost  # 对应Java: private double rentalCost
    
    # Getter和Setter方法
    @property
    def id(self) -> int:
        return self._id
    
    @id.setter
    def id(self, value: int):
        self._id = value
    
    @property
    def port(self) -> str:
        return self._port
    
    @port.setter
    def port(self, value: str):
        self._port = value
    
    @property
    def region(self) -> str:
        return self._region
    
    @region.setter
    def region(self, value: str):
        self._region = value
    
    @property
    def whether_trans(self) -> int:
        return self._whether_trans
    
    @whether_trans.setter
    def whether_trans(self, value: int):
        self._whether_trans = value
    
    @property
    def group(self) -> int:
        return self._group
    
    @group.setter
    def group(self, value: int):
        self._group = value
    
    @property
    def turn_over_time(self) -> int:
        return self._turn_over_time
    
    @turn_over_time.setter
    def turn_over_time(self, value: int):
        self._turn_over_time = value
    
    @property
    def laden_demurrage_cost(self) -> float:
        return self._laden_demurrage_cost
    
    @laden_demurrage_cost.setter
    def laden_demurrage_cost(self, value: float):
        self._laden_demurrage_cost = value

    @property
    def empty_demurrage_cost(self) -> float:
        return self._empty_demurrage_cost
    
    @empty_demurrage_cost.setter
    def empty_demurrage_cost(self, value: float):
        self._empty_demurrage_cost = value
    
    @property
    def loading_cost(self) -> float:
        return self._loading_cost
    
    @loading_cost.setter
    def loading_cost(self, value: float):
        self._loading_cost = value
    
    @property
    def discharge_cost(self) -> float:
        return self._discharge_cost
    
    @discharge_cost.setter
    def discharge_cost(self, value: float):
        self._discharge_cost = value
    
    @property
    def transshipment_cost(self) -> float:
        return self._transshipment_cost
    
    @transshipment_cost.setter
    def transshipment_cost(self, value: float):
        self._transshipment_cost = value
    
    @property
    def rental_cost(self) -> float:
        return self._rental_cost
    
    @rental_cost.setter
    def rental_cost(self, value: float):
        self._rental_cost = value 