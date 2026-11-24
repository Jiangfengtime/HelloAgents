# 本子包提供概念性服务发现与网络接口；便捷函数只是构造 ServiceInfo 并调用内存注册表。
# 当前路径不导入或调用 agent-connect 的加密、认证或网络发送接口，不能视为已具备这些能力。

"""ANP (Agent Network Protocol) 协议实现

概念性实现，提供简洁的 API 用于：
- Agent 服务发现
- Agent 网络管理
- 负载均衡与路由

注意: 这是概念性实现，用于学习和理解 ANP 理念。
详见文档: docs/chapter10/ANP_CONCEPTS.md
"""

from typing import Optional, Dict, Any, List

from .implementation import (
    ANPDiscovery,
    ANPNetwork,
    ServiceInfo
)

def register_service(
    discovery: ANPDiscovery,
    service: Optional[ServiceInfo] = None,
    service_id: Optional[str] = None,
    service_type: Optional[str] = None,
    endpoint: Optional[str] = None,
    service_name: Optional[str] = None,
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """注册服务的便捷函数

    支持两种调用方式：
    1. 传入 ServiceInfo 对象：
       register_service(discovery, service=service_info)

    2. 传入参数自动构造：
       register_service(
           discovery=discovery,
           service_id="agent1",
           service_type="nlp",
           endpoint="http://localhost:8001",
           service_name="NLP Agent",
           capabilities=["text_analysis"],
           metadata={"version": "1.0"}
       )
    """
    # 已有ServiceInfo就直接登记；否则下方用各参数构造对象，两条路最后都写内存表。
    if service is not None:
        # 方式1：直接传入 ServiceInfo 对象
        return discovery.register_service(service)
    else:
        # 方式2：从参数构造 ServiceInfo 对象
        # 只校验三个必填值非空，不验证端点是否存在或是否可连接。
        if not all([service_id, service_type, endpoint]):
            raise ValueError("必须提供 service_id, service_type 和 endpoint 参数")

        service_info = ServiceInfo(
            service_id=service_id,
            service_type=service_type,
            endpoint=endpoint,
            service_name=service_name,
            capabilities=capabilities,
            metadata=metadata
        )
        return discovery.register_service(service_info)

def discover_service(discovery: ANPDiscovery, service_type: str = None):
    """发现服务的便捷函数"""
    return discovery.discover_services(service_type=service_type)

__all__ = [
    "ANPDiscovery",
    "ANPNetwork",
    "ServiceInfo",
    "register_service",
    "discover_service",
]

