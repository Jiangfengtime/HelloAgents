# 固定工具链按预设步骤执行，使用 context 把前一步结果填入后一步模板，不由模型动态规划。
# 链是否应该遇错中止需有显式状态判断；普通错误字符串当前也可能继续进入下一步。

"""工具链管理器 - HelloAgents工具链式调用支持"""

from typing import List, Dict, Any, Optional
from .registry import ToolRegistry


class ToolChain:
    """工具链 - 支持多个工具的顺序执行"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.steps: List[Dict[str, Any]] = []

    def add_step(self, tool_name: str, input_template: str, output_key: str = None):
        """
        添加工具执行步骤
        
        Args:
            tool_name: 工具名称
            input_template: 输入模板，支持变量替换，如 "{input}" 或 "{search_result}"
            output_key: 输出结果的键名，用于后续步骤引用
        """
        # 学习提示：output_key 决定结果保存位置，后续 input_template 可使用该键；同名键会覆盖之前的值。
        step = {
            "tool_name": tool_name,
            "input_template": input_template,
            "output_key": output_key or f"step_{len(self.steps)}_result"
        }
        self.steps.append(step)
        print(f"✅ 工具链 '{self.name}' 添加步骤: {tool_name}")

    def execute(self, registry: ToolRegistry, input_data: str, context: Dict[str, Any] = None) -> str:
        """
        执行工具链
        
        Args:
            registry: 工具注册表
            input_data: 初始输入数据
            context: 执行上下文，用于变量替换
            
        Returns:
            最终执行结果
        """
        # 学习提示：按步骤顺序调用注册表；传入 context 会被原地更新，模板缺键时立即返回错误。
        if not self.steps:
            return "❌ 工具链为空，无法执行"

        print(f"🚀 开始执行工具链: {self.name}")
        
        # 初始化上下文
        if context is None:
            context = {}
        # 复用并修改传入字典，后续步骤通过 output_key 取得前一步结果。
        context["input"] = input_data
        
        final_result = input_data
        
        for i, step in enumerate(self.steps):
            tool_name = step["tool_name"]
            input_template = step["input_template"]
            output_key = step["output_key"]
            
            print(f"📝 执行步骤 {i+1}/{len(self.steps)}: {tool_name}")
            
            # 替换模板中的变量
            try:
                actual_input = input_template.format(**context)
            except KeyError as e:
                return f"❌ 模板变量替换失败: {e}"
            
            # 执行工具
            try:
                result = registry.execute_tool(tool_name, actual_input)
                # 注册表常将异常变成错误字符串，此处仍会向下传递，不能据此判断业务成功。
                context[output_key] = result
                final_result = result
                print(f"✅ 步骤 {i+1} 完成")
            except Exception as e:
                return f"❌ 工具 '{tool_name}' 执行失败: {e}"
        
        print(f"🎉 工具链 '{self.name}' 执行完成")
        return final_result


class ToolChainManager:
    """工具链管理器"""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.chains: Dict[str, ToolChain] = {}

    def register_chain(self, chain: ToolChain):
        """注册工具链"""
        # 学习提示：按链名保存同一对象引用，同名会覆盖，不在注册时验证每个工具是否存在。
        self.chains[chain.name] = chain
        print(f"✅ 工具链 '{chain.name}' 已注册")

    def execute_chain(self, chain_name: str, input_data: str, context: Dict[str, Any] = None) -> str:
        """执行指定的工具链"""
        # 学习提示：找到已登记链后交给 execute；这里不额外判断链返回文本是否代表业务成功。
        if chain_name not in self.chains:
            return f"❌ 工具链 '{chain_name}' 不存在"

        chain = self.chains[chain_name]
        return chain.execute(self.registry, input_data, context)

    def list_chains(self) -> List[str]:
        """列出所有已注册的工具链"""
        return list(self.chains.keys())

    def get_chain_info(self, chain_name: str) -> Optional[Dict[str, Any]]:
        """获取工具链信息"""
        # 学习提示：返回步骤配置用于查看，不运行任何步骤。
        if chain_name not in self.chains:
            return None
        
        chain = self.chains[chain_name]
        return {
            "name": chain.name,
            "description": chain.description,
            "steps": len(chain.steps),
            "step_details": [
                {
                    "tool_name": step["tool_name"],
                    "input_template": step["input_template"],
                    "output_key": step["output_key"]
                }
                for step in chain.steps
            ]
        }


# 便捷函数
def create_research_chain() -> ToolChain:
    """创建一个研究工具链：搜索 -> 计算 -> 总结"""
    # 学习提示：仅构造预设链，示例依赖的工具名需在执行前自行注册。
    chain = ToolChain(
        name="research_and_calculate",
        description="搜索信息并进行相关计算"
    )

    # 步骤1：搜索信息
    chain.add_step(
        tool_name="search",
        input_template="{input}",
        output_key="search_result"
    )

    # 步骤2：基于搜索结果进行计算
    chain.add_step(
        tool_name="my_calculator",
        input_template="2 + 2",  # 简单的计算示例
        output_key="calc_result"
    )

    return chain


def create_simple_chain() -> ToolChain:
    """创建一个简单的工具链示例"""
    # 学习提示：返回配置好的 ToolChain，不会因创建函数被调用就自动执行链。
    chain = ToolChain(
        name="simple_demo",
        description="简单的工具链演示"
    )

    # 只包含一个计算步骤
    chain.add_step(
        tool_name="my_calculator",
        input_template="{input}",
        output_key="result"
    )

    return chain
