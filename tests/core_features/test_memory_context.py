"""真实进程内记忆和上下文流水线；不创建向量库、不下载 tokenizer。"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hello_agents import Message
from hello_agents.memory.base import MemoryConfig, MemoryItem
from hello_agents.memory.types.working import WorkingMemory
from hello_agents.memory.manager import MemoryManager
from hello_agents.context import builder as context_module


@pytest.fixture
def make_memory():
    def make(key, content="alpha beta", user="u1", importance=0.5, minutes_ago=0):
        return MemoryItem(id=key, content=content, memory_type="working", user_id=user,
                          importance=importance, timestamp=datetime.now() - timedelta(minutes=minutes_ago))
    return make


def test_working_memory_crud_and_token_accounting(make_memory):
    memory = WorkingMemory(MemoryConfig())
    assert memory.add(make_memory("a")) == "a"
    assert memory.has_memory("a")
    assert memory.update("a", content="alpha beta gamma", importance=0.9, metadata={"tag": "updated"})
    assert memory.current_tokens == 3
    assert memory.get_all()[0].metadata["tag"] == "updated"
    assert memory.remove("a") is True
    assert memory.current_tokens == 0
    assert memory.remove("a") is False
    assert memory.update("missing", content="x") is False
    memory.add(make_memory("b"))
    memory.clear()
    assert memory.get_all() == [] and memory.current_tokens == 0


def test_working_memory_filters_other_users_and_forgotten(make_memory):
    memory = WorkingMemory(MemoryConfig())
    for key, user in [("mine", "u1"), ("other", "u2"), ("forgotten", "u1")]:
        item = make_memory(key, user=user)
        if key == "forgotten":
            item.metadata["forgotten"] = True
        memory.add(item)
    assert [m.id for m in memory.retrieve("alpha", user_id="u1")] == ["mine"]


def test_working_memory_evicts_lowest_importance(make_memory):
    memory = WorkingMemory(MemoryConfig(working_memory_capacity=2))
    memory.add(make_memory("low", importance=0.1))
    memory.add(make_memory("high", importance=0.9))
    memory.add(make_memory("middle", importance=0.5))
    assert {m.id for m in memory.get_all()} == {"high", "middle"}


def test_working_memory_ttl_cleanup_without_sleep(make_memory):
    memory = WorkingMemory(MemoryConfig(working_memory_ttl_minutes=10))
    memory.add(make_memory("old", minutes_ago=11))
    memory.add(make_memory("new"))  # add 时惰性清理到期项，不等待真实时间流逝。
    assert [m.id for m in memory.get_all()] == ["new"]
    assert memory.get_stats()["current_tokens"] == 2


def test_working_memory_token_capacity(make_memory):
    memory = WorkingMemory(MemoryConfig(working_memory_tokens=3))
    memory.add(make_memory("low", importance=0.1))
    memory.add(make_memory("high", importance=0.9))
    assert [m.id for m in memory.get_all()] == ["high"]
    assert memory.current_tokens <= 3


def test_manager_routes_working_memory_and_user(make_memory):
    manager = MemoryManager(user_id="learner", enable_episodic=False, enable_semantic=False)
    key = manager.add_memory("alpha beta", memory_type="working", importance=0.8, auto_classify=False)
    results = manager.retrieve_memories("alpha", memory_types=["working"])
    assert [m.id for m in results] == [key]
    assert results[0].user_id == "learner"
    with pytest.raises(ValueError, match="不支持"):
        manager.add_memory("x", memory_type="unknown", auto_classify=False)


@pytest.mark.xfail(strict=True, raises=ZeroDivisionError, reason="已知缺陷：空记忆类型列表导致候选配额计算除零")
def test_manager_empty_types_should_return_no_results():
    manager = MemoryManager(enable_episodic=False, enable_semantic=False)
    assert manager.retrieve_memories("alpha", memory_types=[]) == []


@pytest.fixture
def builder_factory(monkeypatch):
    # 使用一字符一计数的确定性替身，避免 tiktoken 首次使用时下载词表。
    # 验证的是上下文选择/结构/截断；不验证真实模型 Token 数。
    monkeypatch.setattr(context_module.tiktoken, "get_encoding", lambda name: SimpleNamespace(encode=list))
    monkeypatch.setattr(context_module, "count_tokens", len)
    return context_module.ContextBuilder


def test_context_retrieval_history_and_structure(builder_factory):
    memory = SimpleNamespace(run=Mock(side_effect=["alpha 任务状态", "alpha 用户偏好"]))
    rag = SimpleNamespace(run=Mock(return_value="alpha 知识证据"))
    builder = builder_factory(memory_tool=memory, rag_tool=rag)
    result = builder.build("alpha", [Message("alpha 历史", "user")], "保持中文")
    for text in ["[Role & Policies]", "保持中文", "[Task]", "alpha 任务状态", "alpha 用户偏好", "alpha 知识证据", "alpha 历史", "[Output]"]:
        assert text in result
    assert memory.run.call_count == 2
    assert rag.run.call_args.args[0]["query"] == "alpha"


def test_context_excludes_unrelated_packets(builder_factory):
    builder = builder_factory()
    packets = [context_module.ContextPacket("alpha 相关", metadata={"type": "tool_result"}),
               context_module.ContextPacket("无关内容", metadata={"type": "tool_result"})]
    result = builder.build("alpha", additional_packets=packets)
    assert "alpha 相关" in result
    assert "无关内容" not in result


def test_context_budget_and_compression(builder_factory):
    config = context_module.ContextConfig(max_tokens=20, reserve_ratio=0.25)
    builder = builder_factory(config=config)
    assert config.get_available_tokens() == 15
    assert builder._compress("first\nsecond\n" + "x" * 20) == "first\nsecond"
    config.enable_compression = False
    assert builder._compress("x" * 100) == "x" * 100


def test_context_keeps_only_recent_ten_messages(builder_factory):
    builder = builder_factory()
    history = [Message(f"alpha turn-{i:02d}", "user") for i in range(12)]
    result = builder.build("alpha", conversation_history=history)
    assert "turn-00" not in result and "turn-01" not in result
    assert "turn-02" in result and "turn-11" in result


def test_context_failed_retrieval_still_includes_task(builder_factory):
    failing = SimpleNamespace(run=Mock(side_effect=RuntimeError("检索故障")))
    result = builder_factory(memory_tool=failing, rag_tool=failing).build("alpha", system_instructions="中文")
    assert "[Task]" in result and "alpha" in result and "中文" in result
