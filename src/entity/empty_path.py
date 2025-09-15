from dataclasses import dataclass
from typing import Optional
from .container_path import ContainerPath
from .port import Port

@dataclass
class EmptyPath:
    """
    空箱路径类
    
    存储空箱路径相关的属性和信息
    
    属性:
        container_path: 集装箱路径对象
        request_id: 请求ID
        origin_port_string: 起始港口字符串
        origin_port: 起始港口对象
        origin_time: 起始时间
        path_id: 路径ID
    """
    
    path_id: int  # 路径ID（建议主流程用container_path_id，path_id仅为兼容）
    request_id: int  # 请求ID
    origin_port_string: str  # 起始港口字符串
    origin_time: int  # 起始时间
    container_path: Optional[ContainerPath] = None  # 集装箱路径对象
    origin_port: Optional[Port] = None  # 起始港口对象


    def __init__(self, path_id: int, request_id: int, origin_port_string: str, origin_time: int, container_path: Optional[ContainerPath] = None, origin_port: Optional[Port] = None):
        self.path_id = path_id
        self.request_id = request_id
        self.origin_port_string = origin_port_string
        self.origin_time = origin_time
        self.container_path = container_path
        self.origin_port = origin_port
