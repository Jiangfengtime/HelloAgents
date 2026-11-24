# 本文件实际维护 Python 字典和邻接表，用于演示注册、筛选与路径计算，没有真实网络传输。
# endpoint 是保存的地址信息；route_message 返回路径、broadcast_message 返回节点名，均不发送 message。

"""
基于 agent-connect 库的 ANP 协议实现

使用 agent-connect 库 (v0.3.7) 实现 Agent Network Protocol 功能。

注意：agent-connect 是一个底层的网络协议库，提供了加密、认证等功能。
这里我们创建一个简化的包装器，使其更易于使用。
"""

from typing import Dict, Any, List, Optional
import asyncio
import json


# 由于 agent-connect 的 API 比较底层，我们创建一个简化的实现
# 实际使用时可以根据需要调用 agent-connect 的具体模块

class ServiceInfo:
    """服务信息"""

    def __init__(
        self,
        service_id: str,
        service_type: str,
        endpoint: str,
        service_name: Optional[str] = None,
        capabilities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # 学习提示：保存服务的元信息与地址，capabilities 只是能力标签，不自动生成可执行处理器。
        self.service_id = service_id
        self.service_type = service_type
        self.endpoint = endpoint
        self.service_name = service_name or service_id
        self.capabilities = capabilities or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        # 学习提示：序列化为普通字典，capabilities/metadata 未深拷贝，修改嵌套值可能影响原对象。
        return {
            "service_id": self.service_id,
            "service_type": self.service_type,
            "endpoint": self.endpoint,
            "service_name": self.service_name,
            "capabilities": self.capabilities,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ServiceInfo':
        """从字典创建"""
        # 学习提示：@classmethod 通过 cls 创建对象；必填键缺失会直接 KeyError，未做完整 schema 校验。
        return cls(
            service_id=data["service_id"],
            service_type=data["service_type"],
            endpoint=data["endpoint"],
            service_name=data.get("service_name"),
            capabilities=data.get("capabilities"),
            metadata=data.get("metadata", {})
        )


class ANPDiscovery:
    """基于 agent-connect 的服务发现实现"""
    
    def __init__(self):
        """初始化服务发现"""
        # 学习提示：所有登记只在当前实例内存中，进程重启后不自动恢复。
        self._services: Dict[str, ServiceInfo] = {}
        
    def register_service(self, service: ServiceInfo) -> bool:
        """
        注册服务
        
        Args:
            service: 服务信息
            
        Returns:
            是否注册成功
        """
        # 学习提示：以 service_id 为键保存对象引用，同 ID 会覆盖；没有端点探活或远程发布。
        self._services[service.service_id] = service
        return True
        
    def unregister_service(self, service_id: str) -> bool:
        """
        注销服务
        
        Args:
            service_id: 服务 ID
            
        Returns:
            是否注销成功
        """
        # 学习提示：仅移除本地登记；不会停止对应 endpoint 上的服务。
        if service_id in self._services:
            del self._services[service_id]
            return True
        return False
        
    def discover_services(
        self,
        service_type: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ServiceInfo]:
        """
        发现服务
        
        Args:
            service_type: 服务类型（可选）
            filters: 过滤条件（可选）
            
        Returns:
            服务列表
        """
        # 学习提示：先按类型过滤，再按 metadata 的键值全匹配；不会按 capabilities 或语义相似度匹配。
        services = list(self._services.values())
        
        # 按类型过滤
        if service_type:
            services = [s for s in services if s.service_type == service_type]
            
        # 按元数据过滤
        if filters:
            def matches_filters(service: ServiceInfo) -> bool:
                # 学习提示：所有过滤键都满足才匹配，是 AND 关系；不解释范围表达式或正则条件。
                for key, value in filters.items():
                    if service.metadata.get(key) != value:
                        return False
                return True
            services = [s for s in services if matches_filters(s)]
            
        return services
        
    def get_service(self, service_id: str) -> Optional[ServiceInfo]:
        """
        获取服务信息
        
        Args:
            service_id: 服务 ID
            
        Returns:
            服务信息，如果不存在则返回 None
        """
        # 学习提示：返回已保存的同一个 ServiceInfo 对象，不是深拷贝。
        return self._services.get(service_id)
        
    def list_all_services(self) -> List[ServiceInfo]:
        """列出所有服务"""
        # 学习提示：仅复制列表容器；列表中的服务对象仍与注册表共享。
        return list(self._services.values())


class ANPNetwork:
    """基于 agent-connect 的网络管理实现"""
    
    def __init__(self, network_id: str = "default"):
        """
        初始化网络管理器
        
        Args:
            network_id: 网络 ID
        """
        # 学习提示：nodes 保存节点信息，connections 保存有向邻接表；不是网络连接池。
        self.network_id = network_id
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._connections: Dict[str, List[str]] = {}
        
    def add_node(self, node_id: str, endpoint: str, metadata: Optional[Dict[str, Any]] = None):
        """
        添加节点到网络
        
        Args:
            node_id: 节点 ID
            endpoint: 节点端点
            metadata: 节点元数据
        """
        # 学习提示：同 ID 会覆盖节点并重置该节点的出边列表；endpoint 只被保存，不会建立连接。
        self._nodes[node_id] = {
            "node_id": node_id,
            "endpoint": endpoint,
            "metadata": metadata or {},
            "status": "active"
        }
        self._connections[node_id] = []
        
    def remove_node(self, node_id: str) -> bool:
        """
        从网络中移除节点
        
        Args:
            node_id: 节点 ID
            
        Returns:
            是否移除成功
        """
        # 学习提示：删除节点以及其他节点指向它的边；不停止 endpoint 对应的外部进程。
        if node_id in self._nodes:
            del self._nodes[node_id]
            del self._connections[node_id]
            # 移除其他节点到此节点的连接
            for connections in self._connections.values():
                if node_id in connections:
                    connections.remove(node_id)
            return True
        return False
        
    def connect_nodes(self, from_node: str, to_node: str):
        """
        连接两个节点
        
        Args:
            from_node: 源节点 ID
            to_node: 目标节点 ID
        """
        # 学习提示：只登记 from→to 的有向边，不自动添加反向边，也没有握手或探活。
        if from_node in self._connections and to_node in self._nodes:
            if to_node not in self._connections[from_node]:
                self._connections[from_node].append(to_node)
                
    def route_message(
        self,
        from_node: str,
        to_node: str,
        message: Dict[str, Any]
    ) -> Optional[List[str]]:
        """
        路由消息（简单的直接路由）
        
        Args:
            from_node: 源节点 ID
            to_node: 目标节点 ID
            message: 消息内容
            
        Returns:
            路由路径，如果无法路由则返回 None
        """
        # 学习提示：只查直达或一个中间节点的路径；未使用 message 发送数据，也不查找任意多跳路径。
        if from_node not in self._nodes or to_node not in self._nodes:
            return None
            
        # 简单实现：直接路由
        if to_node in self._connections.get(from_node, []):
            return [from_node, to_node]
            
        # 尝试通过一跳中转
        for intermediate in self._connections.get(from_node, []):
            if to_node in self._connections.get(intermediate, []):
                return [from_node, intermediate, to_node]
                
        return None
        
    def broadcast_message(self, from_node: str, message: Dict[str, Any]) -> List[str]:
        """
        广播消息到所有连接的节点
        
        Args:
            from_node: 源节点 ID
            message: 消息内容
            
        Returns:
            接收消息的节点列表
        """
        # 学习提示：返回直接邻居的列表副本，message 不被发送；返回接收者名称不能当作送达确认。
        if from_node not in self._connections:
            return []
            
        return self._connections[from_node].copy()
        
    def get_network_stats(self) -> Dict[str, Any]:
        """
        获取网络统计信息
        
        Returns:
            网络统计信息
        """
        # 学习提示：active 来自登记的状态字符串；连接数按有向边统计，不代表真实在线连接数。
        total_connections = sum(len(conns) for conns in self._connections.values())
        active_nodes = sum(1 for node in self._nodes.values() if node["status"] == "active")
        
        return {
            "network_id": self.network_id,
            "total_nodes": len(self._nodes),
            "active_nodes": active_nodes,
            "total_connections": total_connections,
            "nodes": list(self._nodes.keys())
        }
        
    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        获取节点信息
        
        Args:
            node_id: 节点 ID
            
        Returns:
            节点信息，如果不存在则返回 None
        """
        # 学习提示：浅拷贝节点字典并复制邻居列表；嵌套 metadata 仍与内部节点共享。
        if node_id in self._nodes:
            node_info = self._nodes[node_id].copy()
            node_info["connections"] = self._connections[node_id].copy()
            return node_info
        return None


# 示例：创建一个简单的 ANP 网络
def create_example_network() -> ANPNetwork:
    """创建一个示例 ANP 网络"""
    # 学习提示：构造内存拓扑，示例 URL 无需有服务监听；调用这个函数不会部署三个 Agent。
    network = ANPNetwork(network_id="example_network")
    
    # 添加节点
    network.add_node("node1", "http://localhost:8001", {"type": "agent", "role": "coordinator"})
    network.add_node("node2", "http://localhost:8002", {"type": "agent", "role": "worker"})
    network.add_node("node3", "http://localhost:8003", {"type": "agent", "role": "worker"})
    
    # 连接节点
    network.connect_nodes("node1", "node2")
    network.connect_nodes("node1", "node3")
    network.connect_nodes("node2", "node3")
    
    return network


if __name__ == "__main__":
    # 创建示例网络
    network = create_example_network()
    print(f"🌐 ANP Network: {network.network_id}")
    print(f"📊 Network Stats:")
    stats = network.get_network_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    print()
    
    # 测试路由
    print("🔀 Testing message routing:")
    path = network.route_message("node1", "node2", {"type": "test", "content": "Hello"})
    print(f"   Route from node1 to node2: {' -> '.join(path) if path else 'No route found'}")
    
    # 测试广播
    print("\n📢 Testing broadcast:")
    recipients = network.broadcast_message("node1", {"type": "broadcast", "content": "Hello all"})
    print(f"   Broadcast from node1 to: {', '.join(recipients)}")

