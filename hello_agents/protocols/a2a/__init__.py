# A2A 导出层包含历史别名和占位 API；别名存在不代表实现了独立 Agent 或结构化消息协议。
# 实际 implementation.py 用 Flask/requests 构造 HTTP 路由；create_message/parse_message 在此仍直接抛错。

"""A2A (Agent-to-Agent Protocol) 协议实现

基于官方 a2a 库的封装，提供简洁的 API 用于：
- Agent 间消息传递
- 任务委托与协商
- 多 Agent 协作

注意: A2A 功能需要安装官方 SDK: pip install a2a
详见文档: docs/chapter10/A2A_GUIDE.md
"""

# A2A 是可选的，需要安装官方 SDK
try:
    from .implementation import (
        A2AServer,
        A2AClient,
        AgentNetwork,
        AgentRegistry
    )
    __all__ = [
        "A2AServer",
        "A2AClient",
        "AgentNetwork",
        "AgentRegistry",
    ]
except ImportError as e:
    # 如果没有安装依赖，提供占位符
    __all__ = []
    
    class _A2ANotAvailable:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "A2A protocol requires the official 'a2a' library. "
                "Install it with: pip install a2a\n"
                "See docs/chapter10/A2A_GUIDE.md for more information."
            )
    
    A2AServer = _A2ANotAvailable
    A2AClient = _A2ANotAvailable
    AgentNetwork = _A2ANotAvailable
    AgentRegistry = _A2ANotAvailable

# 为了向后兼容，提供别名
A2AAgent = A2AServer
A2AMessage = dict  # 简化的消息类型
MessageType = str

def create_message(*args, **kwargs):
    """创建 A2A 消息（占位符）"""
    # 当前仍是占位入口，即便安装依赖也没有实际构造消息的逻辑。
    raise ImportError("Please install a2a library: pip install a2a")

def parse_message(*args, **kwargs):
    """解析 A2A 消息（占位符）"""
    # 此函数尚未实现消息解析，别名A2AMessage=dict也不会补上该行为。
    raise ImportError("Please install a2a library: pip install a2a")

