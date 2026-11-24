# 阅读顺序：MemoryItem 描述一条记录，MemoryConfig 描述策略，BaseMemory 约定增删改查接口。
# 这里定义的数据不会自动进入大模型上下文；还需管理器检索及应用/工具层回填。

"""记忆系统基础类和配置

按照第8章架构设计的基础组件：
- MemoryItem: 记忆项数据结构
- MemoryConfig: 记忆系统配置
- BaseMemory: 记忆基类
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

class MemoryItem(BaseModel):
    """记忆项数据结构"""
    # id 用于跨存储定位同一条记忆；不是正文的语义向量或用户标识。
    id: str
    content: str
    # 管理器用这个值选择 working/episodic/semantic/perceptual 的具体实现。
    memory_type: str
    # 记录归属，供检索过滤；字段本身不实现更新、删除等操作的权限检查。
    user_id: str
    timestamp: datetime
    # 重要性用于排序/淘汰，不是事实置信度；此字段未声明 0～1 的范围约束。
    importance: float = 0.5
    # 扩展属性，如 session_id、来源、模态；不同类型从这里提取各自需要的字段。
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class MemoryConfig(BaseModel):
    """记忆系统配置"""
    
    # 存储路径
    # 情景/感知记忆在此目录保存 SQLite；Qdrant、Neo4j 的连接另行配置。
    storage_path: str = "./memory_data"
    
    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    # TTL 在添加、检索等入口惰性清理，不是后台定时任务。
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]

class BaseMemory(ABC):
    """记忆基类

    定义所有记忆类型的通用接口和行为
    """

    def __init__(self, config: MemoryConfig, storage_backend=None):
        self.config = config
        # 这里只保存后端引用；具体子类还可能自行初始化数据库，并不统一使用它。
        self.storage = storage_backend
        # 从具体子类名推导类型，例如 WorkingMemory → working，便于通用代码标识实现。
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        """添加记忆项

        Args:
            memory_item: 记忆项对象

        Returns:
            记忆ID
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        """检索相关记忆

        Args:
            query: 查询内容
            limit: 返回数量限制
            **kwargs: 其他检索参数

        Returns:
            相关记忆列表
        """
        pass

    @abstractmethod
    def update(self, memory_id: str, content: str = None,
               importance: float = None, metadata: Dict[str, Any] = None) -> bool:
        """更新记忆

        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据

        Returns:
            是否更新成功
        """
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        """删除记忆

        Args:
            memory_id: 记忆ID

        Returns:
            是否删除成功
        """
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        """检查记忆是否存在

        Args:
            memory_id: 记忆ID

        Returns:
            是否存在
        """
        pass

    @abstractmethod
    def clear(self):
        """清空所有记忆"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息

        Returns:
            统计信息字典
        """
        pass

    def _generate_id(self) -> str:
        """生成记忆ID"""
        import uuid
        return str(uuid.uuid4())

    def _calculate_importance(self, content: str, base_importance: float = 0.5) -> float:
        """计算记忆重要性

        Args:
            content: 记忆内容
            base_importance: 基础重要性

        Returns:
            计算后的重要性分数
        """
        importance = base_importance

        # 基于内容长度
        if len(content) > 100:
            importance += 0.1

        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2

        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()
