"""模型接口：配置优先级、请求参数、响应解析、异常和流式行为，不请求真实服务。"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from hello_agents.core import llm as llm_module
from hello_agents.core.exceptions import HelloAgentsException


@pytest.fixture
def client_factory(monkeypatch):
    create = Mock()
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    constructor = Mock(return_value=client)
    monkeypatch.setattr(llm_module, "OpenAI", constructor)
    # 清除会影响 provider / key / 地址的环境变量，避免读取用户真实配置。
    for key in ["LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL", "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY",
                "MODELSCOPE_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY", "ZHIPU_API_KEY", "GLM_API_KEY",
                "OLLAMA_API_KEY", "OLLAMA_HOST", "VLLM_API_KEY", "VLLM_HOST", "LLM_TIMEOUT"]:
        monkeypatch.delenv(key, raising=False)
    return constructor, create


def test_explicit_custom_config_overrides_environment(client_factory, monkeypatch):
    constructor, _ = client_factory
    monkeypatch.setenv("LLM_MODEL_ID", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "env-test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://env.invalid/v1")
    llm = llm_module.HelloAgentsLLM(model="explicit", api_key="test-key", base_url="https://test.invalid/v1", provider="custom", timeout=7)
    assert llm.model == "explicit"
    constructor.assert_called_once_with(api_key="test-key", base_url="https://test.invalid/v1", timeout=7)


def test_custom_config_can_use_environment(client_factory, monkeypatch):
    monkeypatch.setenv("LLM_MODEL_ID", "env-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://test.invalid/v1")
    llm = llm_module.HelloAgentsLLM(provider="custom")
    assert (llm.model, llm.api_key, llm.base_url) == ("env-model", "test-key", "https://test.invalid/v1")


@pytest.mark.parametrize("key,url", [(None, "https://test.invalid/v1"), ("test-key", None)])
def test_missing_credentials_fail_before_client_creation(client_factory, key, url):
    constructor, _ = client_factory
    with pytest.raises(HelloAgentsException, match="API密钥"):
        llm_module.HelloAgentsLLM(model="test", api_key=key, base_url=url, provider="custom")
    constructor.assert_not_called()


def test_invoke_forwards_parameters_and_returns_text(client_factory):
    _, create = client_factory
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="回答"))])
    llm = llm_module.HelloAgentsLLM(model="test", api_key="test-key", base_url="https://test.invalid/v1", provider="custom")
    messages = [{"role": "user", "content": "问题"}]
    assert llm.invoke(messages, temperature=0.1, max_tokens=12, seed=42) == "回答"
    assert create.call_args.kwargs == {"model": "test", "messages": messages, "temperature": 0.1, "max_tokens": 12, "seed": 42}


def test_invoke_wraps_client_exception(client_factory):
    _, create = client_factory
    create.side_effect = RuntimeError("服务不可用")
    llm = llm_module.HelloAgentsLLM(model="test", api_key="test-key", base_url="https://test.invalid/v1", provider="custom")
    with pytest.raises(HelloAgentsException, match="服务不可用"):
        llm.invoke([{"role": "user", "content": "问题"}])


def test_stream_skips_empty_deltas_and_is_lazy(client_factory):
    _, create = client_factory
    create.return_value = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]) for text in [None, "你", "", "好"]]
    llm = llm_module.HelloAgentsLLM(model="test", api_key="test-key", base_url="https://test.invalid/v1", provider="custom")
    stream = llm.stream_invoke([{"role": "user", "content": "问题"}], temperature=0.3)
    create.assert_not_called()
    assert list(stream) == ["你", "好"]
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["temperature"] == 0.3
