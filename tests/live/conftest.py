"""真实服务夹具：所有模型请求发往 .env 配置的服务，无预设模型回复。"""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest
from dotenv import load_dotenv
from openai import OpenAI

from hello_agents import HelloAgentsLLM

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def pytest_collection_modifyitems(items):
    # 只控制本目录，绝不影响已有离线测试。显式开启后，配置错误应失败而非跳过。
    if os.getenv("HELLO_AGENTS_LIVE_TEST") != "1":
        for item in items:
            if Path(__file__).parent in Path(str(item.path)).parents:
                item.add_marker(pytest.mark.skip(reason="设置 HELLO_AGENTS_LIVE_TEST=1 执行真实模型请求"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    if report.when == "call":
        item.live_test_outcome = report.outcome


class LiveLLM(HelloAgentsLLM):
    """使用真实 OpenAI 兼容客户端，借助 HTTP 事件记录请求与响应。"""

    def __init__(self):
        self.requests = []
        self.responses = []
        self.stream_chunks = []
        # 显式采用统一配置，防止其他服务商环境变量改变 provider 自动识别结果。
        super().__init__(
            provider="custom", model=os.environ["LLM_MODEL_ID"],
            api_key=os.environ["LLM_API_KEY"], base_url=os.environ["LLM_BASE_URL"],
            temperature=0, max_tokens=2048, timeout=60,
        )

    def _create_client(self):
        return OpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout,
            max_retries=0,  # 一个失败直接报告，不因 SDK 自动重试增加调用次数。
            http_client=httpx.Client(timeout=self.timeout, event_hooks={
                "request": [self._on_request], "response": [self._on_response],
            }),
        )

    def _on_request(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        # 不打印 HTTP headers 或 key，只展示本测试生成的请求正文。
        print("\n[真实模型请求]\n" + json.dumps(body, ensure_ascii=False, indent=2))

    def _on_response(self, response):
        body = json.loads(response.request.content)
        if not body.get("stream"):
            response.read()
            try:
                payload = response.json()
            except ValueError:
                payload = {"status": response.status_code, "error": "服务返回非 JSON 内容"}
            # 部分服务商错误正文可能引用凭据，保存与显示前额外脱敏。
            payload = json.loads(json.dumps(payload, ensure_ascii=False).replace(self.api_key, "[REDACTED]"))
            self.responses.append(payload)
            print("\n[真实模型响应]\n" + json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(f"\n[真实流式连接] HTTP {response.status_code}")

    def stream_invoke(self, messages, **kwargs):
        for chunk in super().stream_invoke(messages, **kwargs):
            self.stream_chunks.append(chunk)
            yield chunk


@pytest.fixture
def live_llm(request):
    missing = [name for name in ("LLM_MODEL_ID", "LLM_API_KEY", "LLM_BASE_URL") if not os.getenv(name)]
    if missing:
        pytest.fail("真实调用缺少配置：" + ", ".join(missing))
    llm = LiveLLM()
    host = urlsplit(llm.base_url).hostname
    print(f"\n[真实服务] model={llm.model}, host={host}")
    yield llm
    # 每个用例保留独立记录，方便验证调用消息、工具协议和模型实际输出。
    # 只写测试数据和响应，不保存 API key、请求头或整个 .env。
    output = ROOT / ".local-test-results" / "live"
    try:
        output.mkdir(parents=True, exist_ok=True)
        report = {
            "test": request.node.nodeid, "outcome": getattr(request.node, "live_test_outcome", "unknown"),
            "model": llm.model, "host": host,
            "requests": llm.requests, "responses": llm.responses,
            "stream_chunks": llm.stream_chunks,
        }
        (output / (request.node.name + ".json")).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[调用记录] .local-test-results/live/{request.node.name}.json")
    finally:
        llm._client.close()
