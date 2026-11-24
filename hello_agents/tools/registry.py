# 注册表是进程内的名称查找表，不是工具代码生成器；保存实例/函数引用后由调用方显式执行。
# 类工具和普通函数分开保存，执行优先级、参数形状与错误返回需要分别理解。

"""工具注册表 - HelloAgents原生工具系统"""

from typing import Optional, Any, Callable
from .base import Tool

class ToolRegistry:
    """
    HelloAgents工具注册表

    提供工具的注册、管理和执行功能。
    支持两种工具注册方式：
    1. Tool对象注册（推荐）
    2. 函数直接注册（简便）
    """

    def __init__(self):
        # 类工具和普通函数分别保存；同名查找与执行分支应分开追踪。
        # 以工具名查找实例；例如 {"search": SearchTool对象}。
        self._tools: dict[str, Tool] = {}
        # 普通函数额外保存说明，执行入口接收单个字符串，与类工具的参数字典不同。
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool, auto_expand: bool = True):
        """
        注册Tool对象

        Args:
            tool: Tool实例
            auto_expand: 是否自动展开可展开的工具（默认True）
        """
        # 检查工具是否可展开
        if auto_expand and hasattr(tool, 'expandable') and tool.expandable:
            # 先获得父工具的各动作包装对象；有子工具时会登记子名称并跳过父名称登记。
            expanded_tools = tool.get_expanded_tools()
            if expanded_tools:
                # 注册所有展开的子工具
                for sub_tool in expanded_tools:
                    if sub_tool.name in self._tools:
                        print(f"⚠️ 警告：工具 '{sub_tool.name}' 已存在，将被覆盖。")
                    # 同名赋值会替换旧引用；没有在这里复制或启动工具。
                    self._tools[sub_tool.name] = sub_tool
                print(f"✅ 工具 '{tool.name}' 已展开为 {len(expanded_tools)} 个独立工具")
                return

        # 普通工具或不展开的工具
        if tool.name in self._tools:
            print(f"⚠️ 警告：工具 '{tool.name}' 已存在，将被覆盖。")

        # 保存的是实例引用，不复制对象；共享注册表意味着共享工具内部状态。
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")

    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告：工具 '{name}' 已存在，将被覆盖。")

        # 把函数本身保存在 func 字段，模型给出名称后由程序取出并调用。
        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")

    def unregister(self, name: str):
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        # 只有类工具表没有此名称时才删除函数；两表同名时一次调用只删前者。
        elif name in self._functions:
            del self._functions[name]
            print(f"🗑️ 工具 '{name}' 已注销。")
        else:
            print(f"⚠️ 工具 '{name}' 不存在。")

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取Tool对象"""
        return self._tools.get(name)

    def get_function(self, name: str) -> Optional[Callable]:
        """获取工具函数"""
        func_info = self._functions.get(name)
        return func_info["func"] if func_info else None

    def execute_tool(self, name: str, input_text: str) -> str:
        """
        执行工具

        Args:
            name: 工具名称
            input_text: 输入参数

        Returns:
            工具执行结果
        """
        # 优先查找Tool对象
        # 类工具接收参数字典；普通函数分支接收原始字符串。
        if name in self._tools:
            tool = self._tools[name]
            try:
                # 将字符串包装成固定 input 字段；需要多个业务参数的工具应使用能构造完整参数字典的 Agent 路径。
                return tool.run({"input": input_text})
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        # 查找函数工具
        elif name in self._functions:
            func = self._functions[name]["func"]
            try:
                # 这里才真正执行普通函数；正常返回值或错误文本随后会进入 Agent 的工具反馈。
                return func(input_text)
            except Exception as e:
                return f"错误：执行工具 '{name}' 时发生异常: {str(e)}"

        else:
            return f"错误：未找到名为 '{name}' 的工具。"

    def get_tools_description(self) -> str:
        """
        获取所有可用工具的格式化描述字符串

        Returns:
            工具描述字符串，用于构建提示词
        """
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        # 把每个工具拼成一行，作为提示词中的能力说明；不是原生工具 schema。
        return "\n".join(descriptions) if descriptions else "暂无可用工具"

    def list_tools(self) -> list[str]:
        """列出所有工具名称"""
        return list(self._tools.keys()) + list(self._functions.keys())

    def get_all_tools(self) -> list[Tool]:
        """获取所有Tool对象"""
        return list(self._tools.values())

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._functions.clear()
        print("🧹 所有工具已清空。")

# 该实例在进程内共享；需要会话隔离时显式创建并传入独立 ToolRegistry。
# 全局工具注册表
global_registry = ToolRegistry()
