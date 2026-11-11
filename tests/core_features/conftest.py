"""核心功能测试共用夹具：所有模型回复可控，网络访问立即失败。"""

from copy import deepcopy
import socket
from types import SimpleNamespace

import pytest

from hello_agents import ToolRegistry
from hello_agents.tools.base import Tool, ToolParameter, tool_action


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """记录并阻止网络，避免被测代码捕获异常后把意外联网伪装成测试成功。"""
    attempts = []

    def denied(*args, **kwargs):
        attempts.append("network")
        raise AssertionError("离线测试禁止联网，请注入模拟客户端")

    for name in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, name, denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)
    yield
    assert not attempts, "被测流程尝试了网络访问，即使异常被捕获也不算通过"


class ScriptedLLM:
    """同时支持普通文本、流式文本、原生 tool calls 的模拟模型。"""

    provider = "test"
    model = "offline-model"
    temperature = 0.2
    max_tokens = 100

    def __init__(self, responses=()):
        self.responses = iter(responses)
        self.requests = []
        # FunctionCallAgent 直接走底层客户端；保留这条真实调用路径。
        self._client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create)
        ))

    def _next(self, messages, kwargs):
        self.requests.append({"messages": deepcopy(messages), "kwargs": deepcopy(kwargs)})
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result

    def invoke(self, messages, **kwargs):
        return self._next(messages, kwargs)

    def stream_invoke(self, messages, **kwargs):
        yield from self._next(messages, kwargs)

    def create(self, messages, **kwargs):
        return self._next(messages, kwargs)


class RecordingTool(Tool):
    """有明确参数类型并记录调用的本地工具，可观察转换后的参数。"""

    def __init__(self, name="echo", fail=False):
        super().__init__(name, "返回输入文本")
        self.calls = []
        self.fail = fail

    def get_parameters(self):
        return [
            ToolParameter(name="input", type="string", description="输入"),
            ToolParameter(name="count", type="integer", description="次数", required=False, default=1),
            ToolParameter(name="enabled", type="boolean", description="开关", required=False, default=True),
        ]

    def run(self, parameters):
        self.calls.append(deepcopy(parameters))
        if self.fail:
            raise ValueError("故障工具")
        return parameters.get("input", "") * parameters.get("count", 1)


class ExpandableTool(RecordingTool):
    def __init__(self):
        super().__init__("parent")
        self.expandable = True

    @tool_action("repeat", "重复文本")
    def repeat(self, text: str, count: int = 2) -> str:
        return text * count


@pytest.fixture
def llm_factory():
    return ScriptedLLM


@pytest.fixture
def registry_and_tool():
    registry = ToolRegistry()
    tool = RecordingTool()
    registry.register_tool(tool)
    return registry, tool


@pytest.fixture
def tool_factory():
    return RecordingTool


@pytest.fixture
def expandable_tool():
    return ExpandableTool()


def completion(content=None, calls=()):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=list(calls))
    )])


@pytest.fixture
def response_factory():
    return completion


@pytest.fixture
def call_factory():
    def make(name="echo", arguments='{"input":"你好"}', call_id="call_1"):
        return SimpleNamespace(id=call_id, type="function", function=SimpleNamespace(
            name=name, arguments=arguments
        ))
    return make
