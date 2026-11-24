# 统一入口流程：新增时封装 MemoryItem 并按类型分派；查询时向各类型取候选，再合并排序。
# 管理器不是统一数据库事务层；用户权限、过滤是否落地与跨后端一致性仍需检查具体实现。

"""记忆管理器 - 记忆核心层的统一管理接口"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uuid
import logging

from .base import MemoryItem, MemoryConfig
from .types.working import WorkingMemory
from .types.episodic import EpisodicMemory
from .types.semantic import SemanticMemory
from .types.perceptual import PerceptualMemory
# 存储和检索功能已被各记忆类型内部实现替代

logger = logging.getLogger(__name__)

class MemoryManager:
    """记忆管理器 - 统一的记忆操作接口
    
    负责：
    - 记忆生命周期管理
    - 记忆优先级和重要性评估
    - 记忆遗忘和清理机制
    - 多类型记忆的协调管理
    """
    
    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        user_id: str = "default_user",
        enable_working: bool = True,
        enable_episodic: bool = True,
        enable_semantic: bool = True,
        enable_perceptual: bool = False
    ):
        # 学习提示：enable_* 决定实际构造哪些类型；只学习本地路径时可关闭 episodic/semantic/perceptual。
        self.config = config or MemoryConfig()
        # 一个管理器绑定一个用户；新增记录与检索条件都使用这个标识。
        self.user_id = user_id
        
        # 存储和检索功能已移至各记忆类型内部实现
        
        # 初始化各类型记忆
        # 启用某类型即调用其构造函数；长期记忆后端可能在初始化阶段连接外部数据库。
        self.memory_types = {}
        
        if enable_working:
            self.memory_types['working'] = WorkingMemory(self.config)
        
        if enable_episodic:
            self.memory_types['episodic'] = EpisodicMemory(self.config)
            
        if enable_semantic:
            self.memory_types['semantic'] = SemanticMemory(self.config)
            
        if enable_perceptual:
            self.memory_types['perceptual'] = PerceptualMemory(self.config)
        
        logger.info(f"MemoryManager初始化完成，启用记忆类型: {list(self.memory_types.keys())}")
    
    def add_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_classify: bool = True
    ) -> str:
        """添加记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型
            importance: 重要性分数 (0-1)
            metadata: 元数据
            auto_classify: 是否自动分类到合适的记忆类型
            
        Returns:
            记忆ID
        """
        # 学习提示：auto_classify=True 可覆盖传入类型；要严格写入指定类型时显式关闭自动分类。
        # 自动分类记忆类型
        # auto_classify=True 时可能覆盖调用方传入的 memory_type。
        if auto_classify:
            memory_type = self._classify_memory_type(content, metadata)
        
        # 计算重要性
        if importance is None:
            importance = self._calculate_importance(content, metadata)
        
        # 创建记忆项
        # 统一封装数据再分派：管理器不直接写 SQL，也不直接生成检索向量。
        memory_item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            user_id=self.user_id,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        
        # 添加到对应的记忆类型
        if memory_type in self.memory_types:
            memory_id = self.memory_types[memory_type].add(memory_item)
            logger.debug(f"添加记忆到 {memory_type}: {memory_id}")
            return memory_id
        else:
            raise ValueError(f"不支持的记忆类型: {memory_type}")
    
    def retrieve_memories(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        time_range: Optional[tuple] = None
    ) -> List[MemoryItem]:
        """检索记忆
        
        Args:
            query: 查询内容
            memory_types: 要检索的记忆类型列表
            limit: 返回数量限制
            min_importance: 最小重要性阈值
            time_range: 时间范围 (start_time, end_time)
            
        Returns:
            检索到的记忆列表
        """
        # 学习提示：limit 是最终上限而非保证数量；每类先取固定配额，没有命中的配额不会自动补给其他类型。
        # time_range 虽在签名中声明，当前实现没有将它应用到过滤或传给后端。
        if memory_types is None:
            memory_types = list(self.memory_types.keys())
        
        # 从各个记忆类型中检索
        all_results = []
        # 先为各类型分配候选数量，再合并截断；memory_types 为空会导致除零。
        per_type_limit = max(1, limit // len(memory_types))

        for memory_type in memory_types:
            if memory_type in self.memory_types:
                memory_instance = self.memory_types[memory_type]
                try:
                    # 使用各个记忆类型自己的检索方法
                    # 各类型内部先按自己的相关性规则取候选；后面的汇总会再次按重要性排序。
                    # min_importance 是否生效取决于子类；情景类型当前使用 importance_threshold 名称。
                    type_results = memory_instance.retrieve(
                        query=query,
                        limit=per_type_limit,
                        min_importance=min_importance,
                        user_id=self.user_id
                    )
                    all_results.extend(type_results)
                except Exception as e:
                    # 某类失败后继续查询其他类型；返回空结果不一定意味着数据库里没有数据。
                    logger.warning(f"检索 {memory_type} 记忆时出错: {e}")
                    continue

        # 按重要性和相关性排序
        # 当前跨类型汇总仅按 importance 排序，相关性由各类型检索实现决定。
        all_results.sort(key=lambda x: x.importance, reverse=True)
        return all_results[:limit]
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新内容
            importance: 新重要性
            metadata: 新元数据
            
        Returns:
            是否更新成功
        """
        # 查找记忆所在的类型
        # 此处按 ID 查找，不再检查 user_id；对外暴露更新接口时需额外验证记录归属。
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.update(memory_id, content, importance, metadata)
        
        logger.warning(f"未找到记忆: {memory_id}")
        return False
    
    def remove_memory(self, memory_id: str) -> bool:
        """删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否删除成功
        """
        # 学习提示：通过 ID 在已启用类型中查找，首次命中就返回；这里未增加 user_id 所有权校验。
        for memory_type, memory_instance in self.memory_types.items():
            if memory_instance.has_memory(memory_id):
                return memory_instance.remove(memory_id)
        
        logger.warning(f"未找到记忆: {memory_id}")
        return False
    
    def forget_memories(
        self,
        strategy: str = "importance_based",
        threshold: float = 0.1,
        max_age_days: int = 30
    ) -> int:
        """记忆遗忘机制
        
        Args:
            strategy: 遗忘策略 ("importance_based", "time_based", "capacity_based")
            threshold: 遗忘阈值
            max_age_days: 最大保存天数
            
        Returns:
            遗忘的记忆数量
        """
        total_forgotten = 0
        
        # 统一入口只转发策略；各类型可以选择物理删除或其他遗忘实现。
        for memory_type, memory_instance in self.memory_types.items():
            if hasattr(memory_instance, 'forget'):
                forgotten = memory_instance.forget(strategy, threshold, max_age_days)
                total_forgotten += forgotten

        logger.info(f"记忆遗忘完成: {total_forgotten} 条记忆")
        return total_forgotten

    def consolidate_memories(
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7
    ) -> int:
        """记忆整合 - 将重要的短期记忆转换为长期记忆

        Args:
            from_type: 源记忆类型
            to_type: 目标记忆类型
            importance_threshold: 重要性阈值

        Returns:
            整合的记忆数量
        """
        if from_type not in self.memory_types or to_type not in self.memory_types:
            logger.warning(f"记忆类型不存在: {from_type} -> {to_type}")
            return 0

        # 获取高重要性的源记忆
        source_memory = self.memory_types[from_type]
        target_memory = self.memory_types[to_type]

        # 获取需要整合的记忆
        # 整合是迁移现有记录，不调用模型做摘要；get_all 的范围由源类型实现决定。
        all_memories = source_memory.get_all()
        candidates = [
            m for m in all_memories
            if m.importance >= importance_threshold
        ]

        consolidated_count = 0
        for memory in candidates:
            # 移动到目标记忆类型
            # 注意当前顺序：先删除源记录，再写目标；目标失败时没有事务回滚，存在丢失风险。
            if source_memory.remove(memory.id):
                memory.memory_type = to_type
                # 原地提高重要性，当前没有再次限制到 1.0，也不是模型评估的新分数。
                memory.importance *= 1.1  # 提升重要性
                target_memory.add(memory)
                consolidated_count += 1

        logger.info(f"记忆整合完成: {consolidated_count} 条记忆从 {from_type} 转移到 {to_type}")
        return consolidated_count

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        # 学习提示：统计由各类型自行提供；内存缓存数量和数据库总数可能采用不同口径。
        stats = {
            "user_id": self.user_id,
            "enabled_types": list(self.memory_types.keys()),
            "total_memories": 0,
            "memories_by_type": {},
            "config": {
                "max_capacity": self.config.max_capacity,
                "importance_threshold": self.config.importance_threshold,
                "decay_factor": self.config.decay_factor
            }
        }

        for memory_type, memory_instance in self.memory_types.items():
            type_stats = memory_instance.get_stats()
            stats["memories_by_type"][memory_type] = type_stats
            # 使用count字段（活跃记忆数），而不是total_count（包含已遗忘的）
            stats["total_memories"] += type_stats.get("count", 0)

        return stats

    def clear_all_memories(self):
        """清空所有记忆"""
        # 学习提示：逐类型调用 clear，不是一个统一事务，也不会只因 manager 绑定了用户就自动缩小删除范围。
        for memory_type, memory_instance in self.memory_types.items():
            memory_instance.clear()
        logger.info("所有记忆已清空")




    def _classify_memory_type(self, content: str, metadata: Optional[Dict[str, Any]]) -> str:
        """自动分类记忆类型"""
        # 程序规则分类，不调用大模型；显式 metadata.type 优先于正文关键词。
        if metadata and metadata.get("type"):
            return metadata["type"]
        
        # 简单的分类逻辑，可以扩展为更复杂的分类器
        if self._is_episodic_content(content):
            return "episodic"
        elif self._is_semantic_content(content):
            return "semantic"
        else:
            return "working"
    
    def _is_episodic_content(self, content: str) -> bool:
        """判断是否为情景记忆内容"""
        episodic_keywords = ["昨天", "今天", "明天", "上次", "记得", "发生", "经历"]
        return any(keyword in content for keyword in episodic_keywords)
    
    def _is_semantic_content(self, content: str) -> bool:
        """判断是否为语义记忆内容"""
        semantic_keywords = ["定义", "概念", "规则", "知识", "原理", "方法"]
        return any(keyword in content for keyword in semantic_keywords)
    
    def _calculate_importance(self, content: str, metadata: Optional[Dict[str, Any]]) -> float:
        """计算记忆重要性"""
        # 长度、关键词、priority 的启发式加减分；只有调用方未传 importance 才走这里。
        importance = 0.5  # 基础重要性
        
        # 基于内容长度
        if len(content) > 100:
            importance += 0.1
        
        # 基于关键词
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2
        
        # 基于元数据
        if metadata:
            if metadata.get("priority") == "high":
                importance += 0.3
            elif metadata.get("priority") == "low":
                importance -= 0.2
        
        return max(0.0, min(1.0, importance))
    

    def __str__(self) -> str:
        stats = self.get_memory_stats()
        return f"MemoryManager(user={self.user_id}, total={stats['total_memories']})"
