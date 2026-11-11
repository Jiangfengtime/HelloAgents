"""离线学习实验：从项目根目录运行 python -m examples.learning_trace。

预设模型回复只用来展示程序调用链，不验证真实模型能力或服务连通性。
"""

import json
from copy import deepcopy

from hello_agents import (
    PlanAndSolveAgent,
    ReActAgent,
    ReflectionAgent,
    SimpleAgent,
    ToolRegistry,
)
from hello_agents.tools.builtin.calculator import CalculatorTool


class ScriptedLLM:
    """用预设回复替代远程模型，同时保留每次请求的完整快照。"""

    provider = "offline-script"

    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    def invoke(self, messages, **kwargs):
        # 深拷贝保证后续追加或修改消息不会改变已经记录的请求。
        self.requests.append(deepcopy(messages))
        response = next(self._responses)
        print(f"\n[离线模型请求 {len(self.requests)}]")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        print(f"[预设回复] {response}")
        return response


def show_result(agent, question):
    """先运行，再对比本次模型消息和对象长期持有的对话列表。"""
    print(f"\n{'=' * 24} {type(agent).__name__} {'=' * 24}")
    result = agent.run(question)
    print(f"\n[返回值] {result}")
    print(f"[模型调用次数] {len(agent.llm.requests)}")
    print("[Agent 历史，注意不包含完整中间轨迹]")
    print(json.dumps([m.to_dict() for m in agent.get_history()], ensure_ascii=False, indent=2))


def main():
    # SimpleAgent：回复文本中的标记触发真实的本地计算器，再把结果交回模拟模型。
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    show_result(
        SimpleAgent(
            name="文本工具助手",
            llm=ScriptedLLM(["[TOOL_CALL:python_calculator:2+3]", "结果是5"]),
            tool_registry=registry,
        ),
        "计算2+3",
    )

    # ReAct：第一步执行本地函数，第二次请求会在模板中看到 Observation。
    react_registry = ToolRegistry()
    react_registry.register_function("double", "重复输入文本", lambda text: text * 2)
    show_result(
        ReActAgent(
            name="行动助手",
            llm=ScriptedLLM([
                "Thought: 需要重复文本\nAction: double[你好]",
                "Thought: 已取得结果\nAction: Finish[你好你好]",
            ]),
            tool_registry=react_registry,
        ),
        "把你好重复一次",
    )

    # 一轮反思包含评审和改写，加上初稿，总计三次模型调用。
    show_result(
        ReflectionAgent(
            name="反思助手",
            llm=ScriptedLLM(["Agent能调用工具。", "请补充程序与模型的分工。", "模型提出调用，程序执行工具。"]),
            max_iterations=1,
        ),
        "解释Agent工具调用",
    )

    # Planner 需要严格的 python 围栏，Executor 的最后一步回复直接成为返回值。
    show_result(
        PlanAndSolveAgent(
            name="规划助手",
            llm=ScriptedLLM([
                '```python\n["解释模型职责", "结合前一步解释程序职责"]\n```',
                "模型提出回答或工具请求。",
                "模型提出请求，程序解析并执行已注册工具，再把结果交回模型。",
            ]),
        ),
        "解释模型与程序的分工",
    )


if __name__ == "__main__":
    main()
