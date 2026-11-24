# 类型包同时导出记忆实现与辅助对象（Episode/Entity/Relation/Perception），便于上层直接引用。
# 四类记忆共享 BaseMemory 契约，但使用不同后端；导出名称不能代替服务可用性检查。

"""记忆类型层模块

按照第8章架构设计的记忆类型层：
- WorkingMemory: 工作记忆 - 短期上下文管理
- EpisodicMemory: 情景记忆 - 具体交互事件存储
- SemanticMemory: 语义记忆 - 抽象知识和概念存储
- PerceptualMemory: 感知记忆 - 多模态数据存储
"""

from .working import WorkingMemory
from .episodic import EpisodicMemory, Episode
from .semantic import SemanticMemory, Entity, Relation
from .perceptual import PerceptualMemory, Perception

__all__ = [
    # 记忆类型
    "WorkingMemory",
    "EpisodicMemory",
    "SemanticMemory",
    "PerceptualMemory",

    # 辅助类
    "Episode",
    "Entity",
    "Relation",
    "Perception"
]
