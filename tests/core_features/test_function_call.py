"""保留 FunctionCallAgent 的真实控制流，只替换最外侧模型客户端。"""

import pytest
from hello_agents import FunctionCallAgent


def test_function_call_schema_and_typed_parameters(llm_factory, registry_and_tool, response_factory, call_factory):
    registry, tool = registry_and_tool
    llm = llm_factory([
        response_factory(calls=[call_factory(arguments='{"input":"甲","count":"2","enabled":"false"}')]),
        response_factory("完成"),
    ])
    agent = FunctionCallAgent("助手", llm, tool_registry=registry)
    assert agent.run("调用", temperature=0.4) == "完成"
    assert tool.calls == [{"input": "甲", "count": 2, "enabled": False}]
    request = llm.requests[0]["kwargs"]
    schema = request["tools"][0]["function"]["parameters"]
    assert schema["properties"]["count"]["type"] == "integer"
    assert schema["required"] == ["input"]
    assert request["temperature"] == 0.4
    assert request["model"] == "offline-model"


def test_multiple_calls_match_results_by_id(llm_factory, registry_and_tool, response_factory, call_factory):
    registry, tool = registry_and_tool
    llm = llm_factory([
        response_factory(calls=[call_factory(arguments='{"input":"A"}', call_id="a"),
                                call_factory(arguments='{"input":"B"}', call_id="b")]),
        response_factory("AB"),
    ])
    agent = FunctionCallAgent("助手", llm, tool_registry=registry)
    assert agent.run("同时请求") == "AB"
    messages = llm.requests[1]["messages"]
    assistant = next(m for m in messages if m.get("tool_calls"))
    assert [call["id"] for call in assistant["tool_calls"]] == ["a", "b"]
    assert [(m["tool_call_id"], m["content"]) for m in messages if m["role"] == "tool"] == [("a", "A"), ("b", "B")]
    assert len(tool.calls) == 2
    assert [m.role for m in agent.get_history()] == ["user", "assistant"]


@pytest.mark.parametrize("arguments", ["not-json", "[]", "null", ""])
def test_malformed_arguments_do_not_escape_loop(llm_factory, registry_and_tool, response_factory, call_factory, arguments):
    registry, tool = registry_and_tool
    llm = llm_factory([response_factory(calls=[call_factory(arguments=arguments)]), response_factory("完成")])
    assert FunctionCallAgent("助手", llm, tool_registry=registry).run("异常参数") == "完成"
    assert tool.calls == [{}]  # 当前解析器把非法 JSON / 非字典转为空字典。
    assert llm.requests[1]["messages"][-1]["tool_call_id"] == "call_1"


@pytest.mark.parametrize("limit", [0, 1, 2])
def test_function_call_limit_forces_no_more_tools(llm_factory, registry_and_tool, response_factory, call_factory, limit):
    registry, tool = registry_and_tool
    llm = llm_factory([response_factory(calls=[call_factory()]) for _ in range(limit)] + [response_factory("收尾")])
    agent = FunctionCallAgent("助手", llm, tool_registry=registry)
    assert agent.run("测试", max_tool_iterations=limit) == "收尾"
    assert len(tool.calls) == limit
    assert llm.requests[-1]["kwargs"]["tool_choice"] == "none"
    assert len(llm.requests) == limit + 1


def test_function_registry_schema_and_execution(llm_factory, response_factory, call_factory):
    from hello_agents import ToolRegistry
    registry = ToolRegistry()
    registry.register_function("double", "重复", lambda text: text * 2)
    llm = llm_factory([response_factory(calls=[call_factory("double", '{"input":"好"}')]), response_factory("好好")])
    assert FunctionCallAgent("助手", llm, tool_registry=registry).run("重复") == "好好"
    assert llm.requests[1]["messages"][-1]["content"] == "好好"
    assert llm.requests[0]["kwargs"]["tools"][0]["function"]["parameters"]["required"] == ["input"]


@pytest.mark.parametrize("name,fail,expected", [("missing", False, "未找到工具"), ("echo", True, "故障工具")])
def test_native_tool_errors_are_returned_to_model(llm_factory, tool_factory, response_factory, call_factory, name, fail, expected):
    from hello_agents import ToolRegistry
    registry = ToolRegistry()
    registry.register_tool(tool_factory(fail=fail))
    llm = llm_factory([response_factory(calls=[call_factory(name)]), response_factory("已处理错误")])
    assert FunctionCallAgent("助手", llm, tool_registry=registry).run("请求") == "已处理错误"
    feedback = llm.requests[1]["messages"][-1]
    assert feedback["role"] == "tool"
    assert expected in feedback["content"]


def test_no_registry_uses_plain_text_and_history(llm_factory):
    llm = llm_factory(["第一答", "第二答"])
    agent = FunctionCallAgent("助手", llm)
    assert agent.run("第一问") == "第一答"
    assert agent.run("第二问") == "第二答"
    assert llm.requests[1]["messages"][1:3] == [
        {"role": "user", "content": "第一问"}, {"role": "assistant", "content": "第一答"},
    ]
    assert "tools" not in llm.requests[0]["kwargs"]
