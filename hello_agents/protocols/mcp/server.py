# 服务端主链路：创建 FastMCP → 注册工具/资源/提示模板 → run 启动选定传输 → SDK 分派处理函数。
# 注册函数不等于执行函数；输入 schema 描述参数形状，实际副作用和权限校验仍由业务处理器承担。

"""
基于 fastmcp 库的 MCP 服务器实现

使用 fastmcp 库实现 Model Context Protocol 服务器功能。
fastmcp 是一个快速创建 MCP 服务器的 Python 库。
"""

from typing import Dict, Any, List, Optional, Callable
try:
    from fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "fastmcp is required for MCP server functionality. "
        "Install it with: pip install fastmcp"
    )


class MCPServer:
    """基于 fastmcp 库的 MCP 服务器"""
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None
    ):
        """
        初始化 MCP 服务器
        
        Args:
            name: 服务器名称
            description: 服务器描述
        """
        # 学习提示：创建 SDK 服务对象并保存名称/说明；description 当前只保存在包装对象，未传入 FastMCP 构造器。
        self.mcp = FastMCP(name=name)
        self.name = name
        self.description = description or f"{name} MCP Server"
        
    def add_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加工具到服务器
        
        Args:
            func: 工具函数
            name: 工具名称（可选，默认使用函数名）
            description: 工具描述（可选，默认使用函数文档字符串）
        """
        # 学习提示：把现有函数交给 SDK 装饰器注册，使用签名/docstring 等描述能力；此时不执行函数体。
        # 使用装饰器注册工具
        if name or description:
            self.mcp.tool(name=name, description=description)(func)
        else:
            self.mcp.tool()(func)
        
    def add_resource(
        self,
        func: Callable,
        uri: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加资源到服务器
        
        Args:
            func: 资源处理函数
            uri: 资源 URI（可选）
            name: 资源名称（可选）
            description: 资源描述（可选）
        """
        # 学习提示：当前只转发 uri 与 func，签名中的 name/description 没有继续传给 SDK。
        # 使用装饰器注册资源
        if uri:
            self.mcp.resource(uri)(func)
        else:
            self.mcp.resource()(func)
        
    def add_prompt(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ):
        """
        添加提示词模板到服务器
        
        Args:
            func: 提示词生成函数
            name: 提示词名称（可选）
            description: 提示词描述（可选）
        """
        # 学习提示：注册可按参数生成消息的模板函数；与工具执行、资源读取属于不同能力入口。
        # 使用装饰器注册提示词
        if name or description:
            self.mcp.prompt(name=name, description=description)(func)
        else:
            self.mcp.prompt()(func)
        
    def run(self, transport: str = "stdio", **kwargs):
        """运行服务器

        Args:
            transport: 传输方式 ("stdio", "http", "sse")
            **kwargs: 传输特定的参数
                - host: HTTP 服务器主机（默认 "127.0.0.1"）
                - port: HTTP 服务器端口（默认 8000）
                - 其他 FastMCP.run() 支持的参数

        Examples:
            # Stdio 传输（默认）
            server.run()

            # HTTP 传输
            server.run(transport="http", host="0.0.0.0", port=8081)

            # SSE 传输
            server.run(transport="sse", host="0.0.0.0", port=8081)
        """
        # 学习提示：传输名与 kwargs 原样交给 FastMCP.run；stdio 与 HTTP 的进程、监听和关闭行为由 SDK 实现决定。
        self.mcp.run(transport=transport, **kwargs)
        
    def get_info(self) -> Dict[str, Any]:
        """
        获取服务器信息
        
        Returns:
            服务器信息字典
        """
        # 学习提示：返回本地配置摘要，不发送 initialize，也不表示客户端已经发现工具或协商成功。
        return {
            "name": self.name,
            "description": self.description,
            "protocol": "MCP"
        }


# 便捷的服务器构建器
class MCPServerBuilder:
    """MCP 服务器构建器，提供链式 API"""

    def __init__(self, name: str, description: Optional[str] = None):
        # 学习提示：立即创建同一个服务对象；后续 with_* 都往此对象注册能力。
        self.server = MCPServer(name, description)
        
    def with_tool(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加工具（链式调用）"""
        # 学习提示：注册后返回 self，所以可连续调用 builder.with_tool(...).with_prompt(...)。
        self.server.add_tool(func, name, description)
        return self
        
    def with_resource(self, func: Callable, uri: Optional[str] = None, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加资源（链式调用）"""
        # 学习提示：链式注册资源，实际参数处理仍由 MCPServer.add_resource 决定。
        self.server.add_resource(func, uri, name, description)
        return self
        
    def with_prompt(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None) -> 'MCPServerBuilder':
        """添加提示词（链式调用）"""
        # 学习提示：链式注册提示模板，不触发模型调用。
        self.server.add_prompt(func, name, description)
        return self
        
    def build(self) -> MCPServer:
        """构建服务器"""
        # 学习提示：直接返回已持有的 server，不复制服务，也不启动传输。
        return self.server
        
    def run(self):
        """构建并运行服务器"""
        # 学习提示：这里只采用默认 stdio 启动；若需要传输参数，可先 build 再调用服务的 run。
        self.server.run()


# 示例：创建一个简单的 MCP 服务器
def create_example_server() -> MCPServer:
    """创建一个示例 MCP 服务器"""
    # 学习提示：只构建并注册计算/问候示例；调用方显式 run 后才对外提供服务。
    server = MCPServer(
        name="example-server",
        description="A simple example MCP server with calculator and greeting tools"
    )
    
    # 添加一个简单的计算器工具
    def calculator(expression: str) -> str:
        """计算数学表达式
        
        Args:
            expression: 要计算的数学表达式，例如 "2 + 2" 或 "10 * 5"
        """
        # 学习提示：字符白名单后仍调用 eval，未限制表达式长度或计算复杂度；仅作教学示例，不能视为完整隔离。
        try:
            # 安全的表达式求值（仅支持基本运算）
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expression):
                return f"Error: Invalid characters in expression"
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    server.add_tool(calculator, name="calculator", description="Calculate a mathematical expression")
    
    # 添加一个问候工具
    def greet(name: str) -> str:
        """生成友好的问候语
        
        Args:
            name: 要问候的人的名字
        """
        # 学习提示：直接按输入拼接文本，不请求大模型；返回值由 SDK 转为工具响应。
        return f"Hello, {name}! Welcome to the MCP server example."
    
    server.add_tool(greet, name="greet", description="Generate a friendly greeting")
    
    return server


if __name__ == "__main__":
    # 创建并运行示例服务器
    server = create_example_server()
    print(f"🚀 Starting {server.name}...")
    print(f"📝 {server.description}")
    print(f"🔌 Protocol: MCP")
    print(f"📡 Transport: stdio")
    print()
    server.run()

