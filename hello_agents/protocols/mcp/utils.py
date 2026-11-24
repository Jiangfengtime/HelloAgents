# 这些函数构造应用层字典，方便保存上下文和响应；不是 MCP SDK 的 JSON-RPC 编解码实现。
# 它们不会发送请求、协商能力、执行工具，也不会自动校验内容是否符合某个协议修订版。

"""
MCP 协议工具函数

提供上下文管理、消息解析等辅助功能。
这些函数主要用于处理 MCP 协议的数据结构。
"""

from typing import Dict, Any, List, Optional, Union
import json


def create_context(
    messages: Optional[List[Dict[str, Any]]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    resources: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建 MCP 上下文对象
    
    Args:
        messages: 消息列表
        tools: 工具列表
        resources: 资源列表
        metadata: 元数据
        
    Returns:
        上下文字典
        
    Example:
        >>> context = create_context(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     tools=[{"name": "calculator", "description": "计算器"}]
        ... )
    """
    # 学习提示：非空列表/字典直接放进结果，没有深拷贝；调用方后续修改这些对象可能影响上下文。
    return {
        "messages": messages or [],
        "tools": tools or [],
        "resources": resources or [],
        "metadata": metadata or {}
    }


def parse_context(context: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    解析 MCP 上下文
    
    Args:
        context: 上下文字符串或字典
        
    Returns:
        解析后的上下文字典
        
    Raises:
        ValueError: 如果上下文格式无效
        
    Example:
        >>> context_str = '{"messages": [], "tools": []}'
        >>> parsed = parse_context(context_str)
    """
    # 学习提示：字符串先做 JSON 解析，只检查顶层是 dict；setdefault 会原地补字段，不校验字段值的类型。
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON context: {e}")
    
    if not isinstance(context, dict):
        raise ValueError("Context must be a dictionary or JSON string")
    
    # 确保必需字段存在
    for field in ["messages", "tools", "resources"]:
        context.setdefault(field, [])
    context.setdefault("metadata", {})
    
    return context


def create_error_response(
    error_message: str,
    error_code: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建错误响应
    
    Args:
        error_message: 错误消息
        error_code: 错误代码
        details: 错误详情
        
    Returns:
        错误响应字典
        
    Example:
        >>> error = create_error_response("Tool not found", "TOOL_NOT_FOUND")
    """
    # 学习提示：这是自定义 error 包装，不是自动满足 JSON-RPC 的 id/code/message 错误信封。
    response = {
        "error": {
            "message": error_message,
            "code": error_code or "UNKNOWN_ERROR"
        }
    }
    
    if details:
        response["error"]["details"] = details
    
    return response


def create_success_response(
    data: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建成功响应
    
    Args:
        data: 响应数据
        metadata: 元数据
        
    Returns:
        成功响应字典
        
    Example:
        >>> response = create_success_response({"result": 42})
    """
    # 学习提示：success/data 是应用层约定，不能替代 MCP SDK 规定的工具结果或资源内容结构。
    response = {
        "success": True,
        "data": data
    }
    
    if metadata:
        response["metadata"] = metadata
    
    return response


__all__ = [
    "create_context",
    "parse_context",
    "create_error_response",
    "create_success_response",
]

