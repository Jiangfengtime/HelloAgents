"""Agent 正常流程、跨轮状态、异常和停止条件。运行：pytest tests/core_features/test_agents.py -v。"""

import pytest
from pydantic import ValidationError

from hello_agents import Message, SimpleAgent, ReActAgent, ReflectionAgent, PlanAndSolveAgent


def test_message_rejects_invalid_role():
    """角色拼错时应在消息创建层失败。"""
    with pytest.raises(ValidationError):
        Message("内容", "unknown")


def test_two_turn_history_and_clear(llm_factory):
    llm = llm_factory(["第一答", "第二答", "重新开始"])
    agent = SimpleAgent("助手", llm, system_prompt="中文回答")
    agent.run("第一问")
    assert agent.run("第二问") == "第二答"
    assert llm.requests[1]["messages"] == [
        {"role": "system", "content": "中文回答"},
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
        {"role": "user", "content": "第二问"},
    ]
    copied = agent.get_history()
    copied.clear()
    assert len(agent.get_history()) == 4  # 清空返回的列表不会清空实例历史。
    agent.clear_history()
    agent.run("新问题")
    assert len(llm.requests[2]["messages"]) == 2


def test_model_failure_does_not_append_partial_history(llm_factory):
    agent = SimpleAgent("助手", llm_factory([RuntimeError("模型不可用")]))
    with pytest.raises(RuntimeError, match="模型不可用"):
        agent.run("你好")
    assert agent.get_history() == []


def test_disabled_tools_are_never_executed(llm_factory, registry_and_tool):
    registry, tool = registry_and_tool
    raw = "[TOOL_CALL:echo:你好]"
    agent = SimpleAgent("助手", llm_factory([raw]), tool_registry=registry, enable_tool_calling=False)
    assert agent.run("测试") == raw
    assert tool.calls == []


def test_multiple_text_tool_calls_and_final_history(llm_factory, registry_and_tool):
    registry, tool = registry_and_tool
    llm = llm_factory(["[TOOL_CALL:echo:甲] [TOOL_CALL:echo:乙]", "完成"])
    agent = SimpleAgent("助手", llm, tool_registry=registry)
    assert agent.run("执行两次") == "完成"
    assert tool.calls == [{"input": "甲"}, {"input": "乙"}]
    feedback = llm.requests[1]["messages"][-1]["content"]
    assert "甲" in feedback and "乙" in feedback
    assert [m.content for m in agent.get_history()] == ["执行两次", "完成"]


@pytest.mark.parametrize("limit,expected_calls", [(0, 0), (1, 1), (2, 2)])
def test_text_tool_iteration_limit(llm_factory, registry_and_tool, limit, expected_calls):
    registry, tool = registry_and_tool
    llm = llm_factory(["[TOOL_CALL:echo:继续]"] * limit + ["收尾"])
    agent = SimpleAgent("助手", llm, tool_registry=registry)
    assert agent.run("测试上限", max_tool_iterations=limit) == "收尾"
    assert len(tool.calls) == expected_calls
    assert len(llm.requests) == limit + 1  # 工具上限之后还有一次收尾请求。


@pytest.mark.parametrize("tool_name,fail,expected", [
    ("missing", False, "未找到工具"), ("echo", True, "故障工具"),
])
def test_text_tool_error_is_fed_back(llm_factory, tool_factory, tool_name, fail, expected):
    from hello_agents import ToolRegistry
    registry = ToolRegistry()
    registry.register_tool(tool_factory(fail=fail))
    llm = llm_factory([f"[TOOL_CALL:{tool_name}:测试]", "说明错误"])
    agent = SimpleAgent("助手", llm, tool_registry=registry)
    assert agent.run("执行") == "说明错误"
    assert expected in llm.requests[1]["messages"][-1]["content"]


def test_stream_accumulates_only_after_consumption(llm_factory):
    llm = llm_factory([["你", "好", "！"]])
    agent = SimpleAgent("助手", llm)
    stream = agent.stream_run("问候")
    assert llm.requests == []  # 生成器未迭代前不会调用模型。
    assert next(stream) == "你"
    assert agent.get_history() == []
    assert list(stream) == ["好", "！"]
    assert agent.get_history()[-1].content == "你好！"


@pytest.mark.parametrize("response", ["", "缺少动作", "Thought: 完成\nAction: Finish[答案]"])
def test_react_empty_invalid_and_finish(llm_factory, response):
    agent = ReActAgent("助手", llm_factory([response]), max_steps=1)
    result = agent.run("问题")
    assert result == ("答案" if "Finish" in response else "抱歉，我无法在限定步数内完成这个任务。")
    assert len(agent.get_history()) == 2


def test_react_tool_error_and_step_limit(llm_factory):
    llm = llm_factory(["Thought: 查找\nAction: missing[x]"] * 2)
    agent = ReActAgent("助手", llm, max_steps=2)
    assert "限定步数" in agent.run("问题")
    assert "未找到" in llm.requests[1]["messages"][0]["content"]
    assert len(llm.requests) == 2


def test_reflection_early_stop_and_reset(llm_factory):
    llm = llm_factory(["初稿", "无需改进", "新任务初稿", "无需改进"])
    agent = ReflectionAgent("助手", llm, max_iterations=3)
    assert agent.run("旧任务") == "初稿"
    assert agent.run("新任务") == "新任务初稿"
    assert len(llm.requests) == 4
    assert len(agent.memory.records) == 2  # 本轮轨迹已替换，不累计上一任务。
    assert "旧任务" not in llm.requests[2]["messages"][0]["content"]


def test_reflection_zero_iterations_returns_initial(llm_factory):
    llm = llm_factory(["初稿"])
    assert ReflectionAgent("助手", llm, max_iterations=0).run("任务") == "初稿"
    assert len(llm.requests) == 1


def test_plan_steps_share_results(llm_factory):
    llm = llm_factory(['```python\n["步骤一", "步骤二"]\n```', "中间证据", "最终结论"])
    agent = PlanAndSolveAgent("助手", llm)
    assert agent.run("复杂问题") == "最终结论"
    assert "中间证据" in llm.requests[2]["messages"][0]["content"]
    assert len(llm.requests) == 3  # 一次规划+两次执行，无额外汇总。


@pytest.mark.parametrize("plan", ["没有代码围栏", "```python\nnot python\n```", "```python\n{}\n```", "```python\n[]\n```"])
def test_invalid_plan_stops_before_execution(llm_factory, plan):
    llm = llm_factory([plan])
    agent = PlanAndSolveAgent("助手", llm)
    assert "无法生成有效" in agent.run("复杂问题")
    assert len(llm.requests) == 1
    assert len(agent.get_history()) == 2


@pytest.mark.xfail(strict=True, raises=AttributeError, reason="已知缺陷：SimpleAgent.remove_tool 调用不存在的 unregister_tool")
def test_simple_agent_remove_tool_contract(llm_factory, registry_and_tool):
    registry, _ = registry_and_tool
    agent = SimpleAgent("助手", llm_factory(), tool_registry=registry)
    assert agent.remove_tool("echo") is True
    assert "echo" not in agent.list_tools()


def test_tool_aware_nested_json_and_listener(llm_factory, registry_and_tool):
    from hello_agents import ToolAwareSimpleAgent
    registry, tool = registry_and_tool
    notifications = []
    llm = llm_factory(['[TOOL_CALL:echo:{"input":"[甲:乙]","count":2}]', "完成"])
    agent = ToolAwareSimpleAgent("助手", llm, tool_registry=registry, tool_call_listener=notifications.append)
    assert agent.run("复杂参数") == "完成"
    assert tool.calls == [{"input": "[甲:乙]", "count": 2}]
    assert notifications[0]["tool_name"] == "echo"
    assert notifications[0]["parsed_parameters"]["count"] == 2
    assert "[甲:乙][甲:乙]" in notifications[0]["result"]


def test_tool_aware_listener_failure_keeps_result(llm_factory, registry_and_tool):
    from hello_agents import ToolAwareSimpleAgent
    registry, tool = registry_and_tool

    def broken_listener(event):
        raise RuntimeError("界面回调故障")

    agent = ToolAwareSimpleAgent("助手", llm_factory(["[TOOL_CALL:echo:你好]", "完成"]),
                                 tool_registry=registry, tool_call_listener=broken_listener)
    assert agent.run("调用") == "完成"
    assert tool.calls == [{"input": "你好"}]
