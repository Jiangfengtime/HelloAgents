# 计算器解析表达式 AST 并按允许的节点和运算映射递归计算，不把整个输入直接交给 eval。
# 返回值为计算结果或错误文本；允许表达式仍可能消耗大量资源，未在此实现独立进程隔离。

"""计算器工具"""

import ast
import operator
import math
from typing import Dict, Any

from ..base import Tool

class CalculatorTool(Tool):
    """Python计算器工具"""
    
    # 支持的操作符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }
    
    # 支持的函数
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }
    
    def __init__(self):
        super().__init__(
            name="python_calculator",
            description="执行数学计算。支持基本运算、数学函数等。例如：2+3*4, sqrt(16), sin(pi/2)等。"
        )
    
    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行计算

        Args:
            parameters: 包含input参数的字典

        Returns:
            计算结果
        """
        # 学习提示：接受 input 或 expression；正常结果与失败都以字符串返回，上层需区分错误文本。
        # 支持两种参数格式：input 和 expression
        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "错误：计算表达式不能为空"

        print(f"🧮 正在计算: {expression}")

        try:
            # 解析表达式
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            result_str = str(result)
            print(f"✅ 计算结果: {result_str}")
            return result_str
        except Exception as e:
            error_msg = f"计算失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _eval_node(self, node):
        """递归计算AST节点"""
        # 学习提示：按 AST 节点递归分派，函数名仅从 FUNCTIONS 查找；OPERATORS 中的 ^ 是异或而不是乘方。
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left), 
                self._eval_node(node.right)
            )
        elif isinstance(node, ast.UnaryOp):
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"不支持的函数: {func_name}")
        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"未定义的变量: {node.id}")
        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")
    
    def get_parameters(self):
        """获取工具参数定义"""
        # 学习提示：公开的 schema 只声明 input，expression 是 run 内部兼容入口。
        from ..base import ToolParameter
        return [
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式，支持基本运算和数学函数",
                required=True
            )
        ]

# 便捷函数
def calculate(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式

    Returns:
        计算结果字符串
    """
    # 学习提示：每次创建临时计算器并直接执行，不经过模型或注册表。
    tool = CalculatorTool()
    return tool.run({"input": expression})
