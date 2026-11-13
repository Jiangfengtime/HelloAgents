"""真实大模型功能测试。运行命令见 tests/live/README.md。

不 monkeypatch、不使用模拟 LLM。工具是实际执行的本地 Python 工具。
模型若没有遵循协议或回答错误，用例直接失败，方便定位真实兼容性问题。
"""
import json
from decimal import Decimal
import re
import secrets

from hello_agents import (
    SimpleAgent, FunctionCallAgent, ReActAgent, ReflectionAgent,
    PlanAndSolveAgent, ToolRegistry,
)
from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.builtin.calculator import CalculatorTool


class ObservedCalculator(CalculatorTool):
    """实际执行原计算器，同时记录结果以证明工具确实被调用。"""
    def __init__(self):
        super().__init__()
        self.executions = []

    def run(self, parameters):
        result = super().run(parameters)
        self.executions.append({"parameters": dict(parameters), "result": result})
        print("\n[实际计算器执行] " + json.dumps(self.executions[-1], ensure_ascii=False))
        return result


class PriceLookup(Tool):
    """本地真实价格查询：价格运行时随机生成，模型必须调用工具才能知道。"""
    def __init__(self):
        super().__init__("price_lookup", "查询商品SKU的实际单价，必须使用本工具取得价格")
        self.price = 31 + secrets.randbelow(59)
        self.executions = []

    def get_parameters(self):
        return [ToolParameter(name="sku", type="string", description="商品编号，如 DEMO-001")]

    def run(self, parameters):
        self.executions.append(dict(parameters))
        if parameters.get("sku") != "DEMO-001":
            return json.dumps({"error": "商品不存在"}, ensure_ascii=False)
        result = json.dumps({"sku": "DEMO-001", "price": self.price}, ensure_ascii=False)
        print("\n[实际价格查询] " + result)
        return result


def assert_number(text, expected):
    # 允许“结果为 3,973”等表现形式，但不接受只包含较长数字的一部分。
    normalized = text.replace(",", "").replace("，", "")
    numbers = [Decimal(n) for n in re.findall(r"(?<![\d.])-?\d+(?:\.\d+)?(?![\d.])", normalized)]
    assert Decimal(expected) in numbers, f"期望答案包含完整数值 {expected}，实际：{text}"


def test_live_direct_model(live_llm):
    """服务连通与基础回答：真实调用模型计算，检查结果。"""
    answer = live_llm.invoke([{"role": "user", "content": "计算137乘以29，只输出整数结果。"}])
    assert_number(answer, 3973)
    assert len(live_llm.requests) == 1


def test_live_multi_turn_memory(live_llm):
    """随机暗号在第一轮输入，第二轮只问回忆，检查真实跨轮上下文。"""
    token = "LEARN-" + secrets.token_hex(4)
    agent = SimpleAgent("真实多轮助手", live_llm, system_prompt="请准确记住用户信息，简短回答。")
    agent.run(f"请记住我的学习暗号是 {token}，回复收到即可。")
    answer = agent.run("刚才我告诉你的学习暗号是什么？只输出暗号。")
    assert token in answer
    assert len(agent.get_history()) == 4
    assert token in live_llm.requests[1]["messages"][1]["content"]


def test_live_stream(live_llm):
    """消费真实服务的流式片段，并验证最后保存的历史。"""
    agent = SimpleAgent("流式助手", live_llm, system_prompt="简洁回答。")
    chunks = list(agent.stream_run("请回复：流式调用成功"))
    text = "".join(chunks)
    print("\n[完整流式回答] " + text)
    assert chunks and text.strip()
    assert "流式调用成功" in text
    assert agent.get_history()[-1].content == text
    assert live_llm.requests[0]["stream"] is True


def test_live_simple_text_tool(live_llm):
    """真实模型生成文本工具标记，原计算器执行，再由模型组织回答。"""
    calculator = ObservedCalculator()
    registry = ToolRegistry()
    registry.register_tool(calculator)
    agent = SimpleAgent("文本工具助手", live_llm, tool_registry=registry,
                        system_prompt="数学题必须先使用python_calculator工具，收到执行结果后再回答。")
    answer = agent.run("请使用工具计算137*29。先输出 [TOOL_CALL:python_calculator:137*29]，取得结果后再输出最终数值。", max_tool_iterations=2)
    assert calculator.executions, "真实模型未触发文本工具调用"
    assert calculator.executions[0]["result"] == "3973"
    assert_number(answer, 3973)
    assert len(live_llm.requests) >= 2


def test_live_native_tools_lookup_then_calculate(live_llm):
    """原生工具连续调用：先查随机单价，再计算4件总价，验证ID配对。"""
    price_tool = PriceLookup()
    calculator = ObservedCalculator()
    registry = ToolRegistry()
    registry.register_tool(price_tool)
    registry.register_tool(calculator)
    agent = FunctionCallAgent(
        "原生工具助手", live_llm, tool_registry=registry, max_tool_iterations=4,
        system_prompt="必须先调用price_lookup查询商品实际单价，再调用python_calculator计算总价，最后回答。不要猜价格。",
    )
    answer = agent.run("查询DEMO-001的单价，买4件总价是多少？必须实际调用两个工具。")
    assert price_tool.executions, "真实模型未调用价格查询工具"
    assert calculator.executions, "真实模型未调用计算器"
    assert any(item["result"] == str(price_tool.price * 4) for item in calculator.executions)
    assert_number(answer, price_tool.price * 4)
    # 检查发往真实服务的工具结果均与此前 assistant tool_calls 的 ID 对应。
    messages = live_llm.requests[-1]["messages"]
    requested = {call["id"] for m in messages for call in m.get("tool_calls", [])}
    returned = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert requested and returned == requested


def test_live_react(live_llm):
    """真实模型按ReAct协议生成Action，实际计算，并基于Observation结束。"""
    calculator = ObservedCalculator()
    registry = ToolRegistry()
    registry.register_tool(calculator)
    agent = ReActAgent("真实ReAct", live_llm, tool_registry=registry, max_steps=4)
    answer = agent.run("必须先调用python_calculator计算137*29，取得Observation后再用Finish返回最终数值。")
    assert calculator.executions, "真实模型未输出可解析的计算器Action"
    assert_number(answer, 3973)
    assert any("Observation:" in row for row in agent.current_history)


def test_live_reflection(live_llm):
    """初稿、真实模型评审、必要时改写；允许真实评审选择提前结束。"""
    agent = ReflectionAgent("真实反思", live_llm, max_iterations=1)
    answer = agent.run("用中文解释Agent工具调用，必须明确提到模型提出工具请求、Python程序执行工具。控制在120字内。")
    assert isinstance(answer, str) and answer.strip()
    assert "模型" in answer and "工具" in answer
    records = agent.memory.records
    assert records[0]["type"] == "execution"
    assert any(row["type"] == "reflection" and row["content"].strip() for row in records)
    assert 2 <= len(live_llm.requests) <= 3


def test_live_plan_and_solve(live_llm):
    """真实模型规划两步，执行器逐步推理并返回最后一步答案。"""
    agent = PlanAndSolveAgent("真实规划", live_llm, custom_prompts={
        "planner": '把以下问题拆为恰好两个步骤。只输出严格的python代码围栏，内容为包含两个字符串的Python列表。问题：{question}',
    })
    answer = agent.run("有3盒铅笔，每盒12支，送出5支后剩多少支？先计算总数，再计算剩余。")
    assert_number(answer, 31)
    assert len(live_llm.requests) == 3, "预期一次真实规划及两次真实步骤执行"
    assert len(agent.get_history()) == 2


def test_live_memory_context_answer(live_llm):
    """真实工作记忆检索→ContextBuilder→真实模型回答，验证证据进入回答。"""
    from hello_agents.context import ContextBuilder, ContextConfig
    from hello_agents.memory.manager import MemoryManager
    from hello_agents.context import ContextPacket
    token = "COURSE-" + secrets.token_hex(4)
    manager = MemoryManager(enable_episodic=False, enable_semantic=False)
    manager.add_memory(f"course 学习项目代号是 {token}", importance=0.9, auto_classify=False)
    memories = manager.retrieve_memories("course", memory_types=["working"])
    assert memories
    packets = [ContextPacket(m.content, metadata={"type": "related_memory"}) for m in memories]
    context = ContextBuilder(config=ContextConfig(max_tokens=2000)).build(
        user_query="course", system_instructions="请依据Evidence回答学习项目的代号，保留完整代号。",
        additional_packets=packets,
    )
    assert token in context
    answer = SimpleAgent("证据助手", live_llm).run(context)
    assert token in answer
