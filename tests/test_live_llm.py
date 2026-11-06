"""真实模型集成测试；显式设置 HELLO_AGENTS_LIVE_TEST=1 才会请求服务。"""
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# 指定路径，避免从 PyCharm 或其他工作目录运行时找不到配置。
load_dotenv(Path(__file__).resolve().parents[1] / '.env')

pytestmark = pytest.mark.skipif(
    os.getenv('HELLO_AGENTS_LIVE_TEST') != '1',
    reason='真实模型测试默认跳过；设置 HELLO_AGENTS_LIVE_TEST=1 执行',
)


def test_live_model_response():
    """直接请求模型，确认服务返回非空文本。"""
    from hello_agents import HelloAgentsLLM

    llm = HelloAgentsLLM()
    result = llm.invoke([{'role': 'user', 'content': '请简短问候。'}])
    assert isinstance(result, str) and result.strip()


def test_live_simple_agent():
    """验证用户问题经 SimpleAgent 到真实模型，再返回回答和历史记录。"""
    from hello_agents import HelloAgentsLLM, SimpleAgent

    agent = SimpleAgent(name='测试助手', llm=HelloAgentsLLM())
    result = agent.run('请用一句话问候我。')
    assert isinstance(result, str) and result.strip()
    history = agent.get_history()
    assert [message.role for message in history] == ['user', 'assistant']
    assert history[-1].content == result
