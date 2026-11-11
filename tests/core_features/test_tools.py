"""计算器、注册表、装饰器展开、工具链和异步包装。"""

import asyncio
import pytest

from hello_agents import ToolRegistry
from hello_agents.tools.builtin.calculator import CalculatorTool
from hello_agents.tools.chain import ToolChain
from hello_agents.tools.async_executor import AsyncToolExecutor


@pytest.mark.parametrize("expression,expected", [
    ("2+3*4", "14"), ("(2+3)*4", "20"), ("-3+1", "-2"),
    ("2**3", "8"), ("sqrt(16)", "4.0"), ("round(3.14159,2)", "3.14"),
])
def test_calculator_valid_expressions(expression, expected):
    assert CalculatorTool().run({"input": expression}) == expected


@pytest.mark.parametrize("expression", ["1/0", "2+", "unknown+1", "__import__('os')", "[x for x in range(2)]"])
def test_calculator_invalid_expressions_return_errors(expression):
    assert "计算失败" in CalculatorTool().run({"input": expression})


def test_calculator_empty_and_expression_alias():
    tool = CalculatorTool()
    assert "不能为空" in tool.run({})
    assert tool.run({"expression": "6*7"}) == "42"


def test_registry_replace_unregister_and_clear():
    registry = ToolRegistry()
    registry.register_function("echo", "旧实现", lambda text: "old")
    registry.register_function("echo", "新实现", lambda text: "new")
    assert registry.execute_tool("echo", "x") == "new"
    registry.unregister("echo")
    assert "未找到" in registry.execute_tool("echo", "x")
    registry.register_function("other", "其他", str)
    registry.clear()
    assert registry.list_tools() == []


def test_registry_class_precedes_function_with_same_name(registry_and_tool):
    registry, tool = registry_and_tool
    registry.register_function("echo", "同名函数", lambda text: "function")
    assert registry.execute_tool("echo", "class") == "class"
    assert tool.calls == [{"input": "class"}]


def test_registry_catches_function_and_tool_errors(tool_factory):
    registry = ToolRegistry()
    registry.register_tool(tool_factory(fail=True))

    def broken(text):
        raise ValueError("函数故障")

    registry.register_function("broken", "抛异常", broken)
    assert "故障工具" in registry.execute_tool("echo", "x")
    assert "函数故障" in registry.execute_tool("broken", "x")


def test_tool_action_expansion_defaults_and_schema(expandable_tool):
    registry = ToolRegistry()
    registry.register_tool(expandable_tool)
    assert registry.list_tools() == ["repeat"]
    subtool = registry.get_tool("repeat")
    assert subtool.run({"text": "好"}) == "好好"
    assert subtool.run({"text": "好", "count": 3}) == "好好好"
    schema = subtool.to_openai_schema()["function"]["parameters"]
    assert schema["required"] == ["text"]
    assert schema["properties"]["count"]["type"] == "integer"
    assert not subtool.validate_parameters({})
    with pytest.raises(TypeError):
        subtool.run({})


def test_tool_can_be_registered_without_expansion(expandable_tool):
    registry = ToolRegistry()
    registry.register_tool(expandable_tool, auto_expand=False)
    assert registry.list_tools() == ["parent"]


def test_chain_transfers_results_and_context():
    registry = ToolRegistry()
    registry.register_function("double", "重复", lambda text: text * 2)
    registry.register_function("label", "加前缀", lambda text: f"结果:{text}")
    chain = ToolChain("示例", "连续处理")
    chain.add_step("double", "{input}", "doubled")
    chain.add_step("label", "{doubled}", "answer")
    context = {}
    assert chain.execute(registry, "好", context) == "结果:好好"
    assert context == {"input": "好", "doubled": "好好", "answer": "结果:好好"}


def test_chain_empty_and_missing_variable_stop():
    registry = ToolRegistry()
    called = []
    registry.register_function("echo", "记录", lambda text: called.append(text) or text)
    chain = ToolChain("示例", "失败分支")
    assert "为空" in chain.execute(registry, "x")
    chain.add_step("echo", "{missing}")
    assert "模板变量替换失败" in chain.execute(registry, "x")
    assert called == []


def test_async_batch_returns_original_task_mapping():
    registry = ToolRegistry()
    registry.register_function("double", "重复", lambda text: text * 2)
    # 不依赖 pytest-asyncio 插件；线程池在 finally 内关闭。
    executor = AsyncToolExecutor(registry, max_workers=2)
    try:
        results = asyncio.run(executor.execute_tools_batch("double", ["A", "B"]))
    finally:
        executor.close()
    assert [(r["task_id"], r["input_data"], r["result"]) for r in results] == [(0, "A", "AA"), (1, "B", "BB")]


@pytest.mark.xfail(strict=True, reason="已知缺陷：链把注册表的错误字符串当作成功结果继续执行")
def test_chain_should_stop_after_failed_tool():
    registry = ToolRegistry()
    following_calls = []
    registry.register_function("next", "后续动作", lambda text: following_calls.append(text) or text)
    chain = ToolChain("错误中止", "错误不应继续传递")
    chain.add_step("missing", "{input}", "first")
    chain.add_step("next", "{first}")
    chain.execute(registry, "x")
    assert following_calls == []
