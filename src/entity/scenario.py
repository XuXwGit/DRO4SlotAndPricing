"""
@Author: XuXw
@Description: 场景类，对应Java版本Scenario.java
@DateTime: 2024/12/4 21:54
"""
from typing import List, Optional
import numpy as np

class Scenario:
    """
    场景类，表示一组需求场景
    对应Java类: multi.Scenario
    """
    
    def __init__(self, request: Optional[List[float]] = None, id: int = 0, scenario_id: int = 0):
        """
        初始化场景对象
        
        支持三种构造方式，对应Java:
        - public Scenario()
        - public Scenario(double[] request)
        - public Scenario(double[] request, int id)
        
        Args:
            request: 需求数组，默认为None
            id: 场景ID，默认为0
            scenario_id: 与id参数等效，为了兼容已有代码，默认为0
        """
        # 基本属性
        self._id: int = id if id != 0 else scenario_id  # 使用id或scenario_id
        self._request: List[float] = [] if request is None else request  # 对应Java: private double[] request
        self._worse_request_set: Optional[List[int]] = None  # 对应Java: private List<Integer> worseRequestSet
    
    def __post_init__(self):
        # 确保request被初始化为空列表
        if self.request is None:
            self.request = []

    def __hash__(self):
        """计算场景哈希值，用于判断场景是否重复"""
        return hash(tuple(self.request))
    
    def __eq__(self, other):
        """比较两个场景是否相等"""
        if not isinstance(other, Scenario):
            return False
        return self.request == other.request
    
    def __str__(self):
        """返回场景的字符串表示"""
        return f"Scenario(requests={self.request})"

    # Getter和Setter方法
    @property
    def id(self) -> int:
        """
        获取场景ID
        对应Java: getId() 使用@Getter注解自动生成
        """
        return self._id
    
    @id.setter
    def id(self, value: int):
        """
        设置场景ID
        对应Java中的setter方法
        """
        self._id = value
    
    @property
    def request(self) -> List[float]:
        """
        获取需求数组
        对应Java: public double[] getRequest()
        """
        return self._request
    
    @request.setter
    def request(self, value: List[float]):
        """
        设置需求数组
        对应Java: public void setRequest(double[] request)
        """
        self._request = value
    
    @property
    def worse_request_set(self) -> Optional[List[int]]:
        """
        获取更差需求集合
        对应Java: public List<Integer> getWorseRequestSet()
        """
        return self._worse_request_set
    
    @worse_request_set.setter
    def worse_request_set(self, value: List[int]):
        """
        设置更差需求集合
        对应Java: public void setWorseRequestSet(List<Integer> worseRequestSet)
        """
        self._worse_request_set = value
    
    def __eq__(self, other) -> bool:
        """
        判断两个场景是否相等
        对应Java: @Override public boolean equals(Object o)
        
        Args:
            other: 要比较的对象
            
        Returns:
            bool: 如果相等返回True，否则返回False
        """
        if self is other:
            return True
        if not isinstance(other, Scenario):
            return False
        
        # 使用numpy的array_equal比较request数组
        requests_equal = np.array_equal(self._request, other._request)
        
        # 比较worse_request_set
        worse_sets_equal = self._worse_request_set == other._worse_request_set
        
        return requests_equal and worse_sets_equal
    
    def __hash__(self) -> int:
        """
        计算场景对象的哈希值
        对应Java: @Override public int hashCode()
        
        Returns:
            int: 哈希值
        """
        # 使用元组包装request数组，使其可哈希
        request_hash = hash(tuple(self._request)) if self._request else 0
        
        # 计算worse_request_set的哈希值
        worse_set_hash = hash(tuple(self._worse_request_set)) if self._worse_request_set else 0
        
        # 组合哈希值
        return hash((request_hash, worse_set_hash)) 