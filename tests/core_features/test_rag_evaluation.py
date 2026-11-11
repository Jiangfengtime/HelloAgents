"""RAG 入库/查询边界与本地评估：替换嵌入服务、向量后端和数据集。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hello_agents import SimpleAgent
from hello_agents.memory.rag import pipeline
from hello_agents.memory.rag.document import Document, DocumentProcessor
from hello_agents.evaluation.benchmarks.gaia.evaluator import GAIAEvaluator


def test_document_chunks_keep_source_and_overlap():
    document = Document("ABCDEFGHIJKL", {"source": "lesson.txt"})
    chunks = DocumentProcessor(chunk_size=6, chunk_overlap=2).process_document(document)
    assert [c.content for c in chunks] == ["ABCDEF", "EFGHIJ", "IJKL"]
    assert all(c.doc_id == document.doc_id for c in chunks)
    assert all(c.metadata["source"] == "lesson.txt" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == 3
    assert "total_chunks" not in document.metadata  # 处理分块不污染原元数据。


def test_loader_missing_empty_and_duplicate_documents(tmp_path, monkeypatch):
    first = tmp_path / "first.md"
    duplicate = tmp_path / "duplicate.md"
    empty = tmp_path / "empty.md"
    for path in [first, duplicate]:
        path.write_text("# Alpha\n\nAlpha evidence.", encoding="utf-8")
    empty.write_text("", encoding="utf-8")
    # 只替换格式转换，真实执行分段、分块、去重和元数据生成。
    monkeypatch.setattr(pipeline, "_convert_to_markdown", lambda path: Path(path).read_text(encoding="utf-8"))
    chunks = pipeline.load_and_chunk_texts([str(first), str(duplicate), str(empty), str(tmp_path / "missing.md")], namespace="lesson")
    assert chunks
    assert len({c["content"] for c in chunks}) == len(chunks)
    assert all(c["metadata"]["namespace"] == "lesson" for c in chunks)
    assert all(c["metadata"]["source_path"] == str(first) for c in chunks)


@pytest.fixture
def embedding_stub(monkeypatch):
    embedder = SimpleNamespace(encode=Mock(side_effect=lambda texts: [[1.0, 0.0] for _ in texts]))
    monkeypatch.setattr(pipeline, "get_text_embedder", lambda: embedder)
    monkeypatch.setattr(pipeline, "get_dimension", lambda default=384: 2)
    return embedder


def test_index_batches_preserve_ids_content_and_namespace(embedding_stub):
    chunks = [{"id": str(i), "content": f"evidence {i}", "metadata": {"source_path": "lesson.md"}} for i in range(3)]
    store = SimpleNamespace(add_vectors=Mock(return_value=True))
    pipeline.index_chunks(store=store, chunks=chunks, batch_size=2, rag_namespace="lesson")
    assert [len(call.args[0]) for call in embedding_stub.encode.call_args_list] == [2, 1]
    payload = store.add_vectors.call_args.kwargs
    assert payload["ids"] == ["0", "1", "2"]
    assert payload["vectors"] == [[1.0, 0.0]] * 3
    assert [m["content"] for m in payload["metadata"]] == ["evidence 0", "evidence 1", "evidence 2"]
    assert all(m["rag_namespace"] == "lesson" and m["is_rag_data"] for m in payload["metadata"])


def test_index_empty_and_storage_failure(embedding_stub):
    store = SimpleNamespace(add_vectors=Mock(return_value=False))
    pipeline.index_chunks(store=store, chunks=[])
    store.add_vectors.assert_not_called()
    embedding_stub.encode.assert_not_called()
    with pytest.raises(RuntimeError, match="Failed to index"):
        pipeline.index_chunks(store=store, chunks=[{"id": "a", "content": "evidence"}])


def test_search_forwards_namespace_threshold_and_limit(monkeypatch):
    monkeypatch.setattr(pipeline, "embed_query", Mock(return_value=[1.0, 0.0]))
    hits = [{"id": "a", "score": 0.9, "metadata": {"content": "证据"}}]
    store = SimpleNamespace(search_similar=Mock(return_value=hits))
    assert pipeline.search_vectors(store, "alpha", top_k=3, rag_namespace="lesson", score_threshold=0.6) == hits
    assert store.search_similar.call_args.kwargs == {
        "query_vector": [1.0, 0.0], "limit": 3, "score_threshold": 0.6,
        "where": {"memory_type": "rag_chunk", "is_rag_data": True, "data_source": "rag_pipeline", "rag_namespace": "lesson"},
    }


def test_search_empty_and_backend_failure(monkeypatch):
    embed = Mock(return_value=[1.0, 0.0])
    monkeypatch.setattr(pipeline, "embed_query", embed)
    store = SimpleNamespace(search_similar=Mock(side_effect=RuntimeError("向量库故障")))
    assert pipeline.search_vectors(store, "") == []
    embed.assert_not_called()
    assert pipeline.search_vectors(store, "alpha") == []  # 当前后端异常降级为空结果。


def test_rank_weights_and_snippet_budget():
    hits = [
        {"id": "a", "score": 0.9, "metadata": {"content": "AAAA"}},
        {"id": "b", "score": 0.5, "metadata": {"content": "BBBB"}},
    ]
    ranked = pipeline.rank(hits, {"b": 1.0}, w_vector=0.5, w_graph=0.5)
    assert [r["memory_id"] for r in ranked] == ["b", "a"]
    assert ranked[0]["score"] == pytest.approx(0.75)
    # max_chars 统计内容，不计片段间连接换行。
    assert pipeline.merge_snippets(ranked, max_chars=6) == "BBBB\n\nAA"


def test_local_index_to_search_roundtrip(monkeypatch, embedding_stub):
    """真实执行 pipeline 入库和过滤调用；替身存储不代表已验证 Qdrant。"""
    records = []

    def add_vectors(vectors, metadata, ids):
        records.extend({"id": key, "metadata": meta, "score": 1.0} for key, meta in zip(ids, metadata))
        return True

    def search_similar(query_vector, limit, score_threshold, where):
        return [r for r in records if all(r["metadata"].get(k) == v for k, v in where.items())][:limit]

    store = SimpleNamespace(add_vectors=add_vectors, search_similar=search_similar)
    monkeypatch.setattr(pipeline, "embed_query", lambda query: [1.0, 0.0])
    for namespace in ["mine", "other"]:
        pipeline.index_chunks(store=store, chunks=[{"id": namespace, "content": namespace + " evidence"}], rag_namespace=namespace)
    hits = pipeline.search_vectors(store, "evidence", rag_namespace="mine")
    assert [h["id"] for h in hits] == ["mine"]


@pytest.mark.parametrize("response,expected_score", [
    ("FINAL ANSWER: [Paris]", 1.0), ("FINAL ANSWER: Paris France", 0.5),
    ("FINAL ANSWER: Berlin", 0.0),
])
def test_gaia_local_sample_scores(llm_factory, response, expected_score):
    evaluator = GAIAEvaluator(dataset=SimpleNamespace(load=lambda: []))
    agent = SimpleAgent("离线评估", llm_factory([response]))
    result = evaluator.evaluate_sample(agent, {"question": "capital?", "final_answer": "Paris", "task_id": "local-1", "level": 1})
    assert result["score"] == expected_score
    assert result["expected"] == "Paris" and result["task_id"] == "local-1"


def test_gaia_exception_records_failure(llm_factory):
    evaluator = GAIAEvaluator(dataset=SimpleNamespace(load=lambda: []))
    result = evaluator.evaluate_sample(SimpleAgent("离线", llm_factory([RuntimeError("模型故障")])), {"question": "q", "final_answer": "Paris"})
    assert result["score"] == 0.0 and result["exact_match"] is False
    assert "模型故障" in result["error"]


@pytest.fixture
def rag_fixture(monkeypatch, tmp_path, llm_factory):
    from hello_agents.tools.builtin import rag_tool as module
    hits = [{"id": "a", "score": 0.95, "metadata": {"content": "RAG先检索资料，再提供证据给模型。", "source_path": "lesson.md"}}]
    search = Mock(return_value=hits)
    search_advanced = Mock(return_value=hits)
    fake_pipeline = {"search": search, "search_advanced": search_advanced, "namespace": "lesson"}
    factory = Mock(return_value=fake_pipeline)
    llm = llm_factory(["依据资料，RAG通过检索增强回答。"])
    monkeypatch.setattr(module, "create_rag_pipeline", factory)
    monkeypatch.setattr(module, "HelloAgentsLLM", lambda: llm)
    rag = module.RAGTool(knowledge_base_path=str(tmp_path / "kb"), rag_namespace="lesson")
    return rag, llm, search, search_advanced


@pytest.mark.parametrize("advanced", [False, True])
def test_rag_ask_injects_evidence_and_sources(rag_fixture, advanced):
    rag, llm, search, search_advanced = rag_fixture
    answer = rag.run({"action": "ask", "query": "RAG是什么", "namespace": "lesson", "enable_advanced_search": advanced})
    assert "通过检索增强回答" in answer and "lesson.md" in answer
    messages = llm.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "RAG先检索资料" in messages[1]["content"]
    assert "RAG是什么" in messages[1]["content"]
    if advanced:
        search.assert_not_called()
        assert search_advanced.call_args.kwargs["enable_mqe"] is True
        assert search_advanced.call_args.kwargs["enable_hyde"] is True
    else:
        search_advanced.assert_not_called()
        assert search.call_args.kwargs == {"query": "RAG是什么", "top_k": 5}


def test_rag_empty_question_and_no_hits_do_not_generate(rag_fixture):
    rag, llm, search, _ = rag_fixture
    assert "请提供" in rag.run({"action": "ask", "question": " "})
    search.return_value = []
    result = rag.run({"action": "ask", "question": "没有资料", "namespace": "lesson", "enable_advanced_search": False})
    assert "没有找到" in result
    assert llm.requests == []


def test_rag_empty_model_answer_is_error(rag_fixture):
    rag, llm, _, _ = rag_fixture
    llm.responses = iter([" "])
    assert "未能生成有效答案" in rag.run({"action": "ask", "question": "问题", "namespace": "lesson"})


def test_rag_initialization_failure_is_visible(monkeypatch, tmp_path):
    from hello_agents.tools.builtin import rag_tool as module
    monkeypatch.setattr(module, "create_rag_pipeline", Mock(side_effect=RuntimeError("存储不可用")))
    rag = module.RAGTool(knowledge_base_path=str(tmp_path))
    assert rag.initialized is False
    assert "存储不可用" in rag.run({"action": "ask", "question": "问题"})


def test_rag_unsupported_action_and_clear_without_confirm(rag_fixture):
    rag, llm, search, advanced = rag_fixture
    assert "不支持的操作" in rag.run({"action": "unknown"})
    assert "confirm=true" in rag.run({"action": "clear", "namespace": "lesson"})
    assert llm.requests == []
    search.assert_not_called()
    advanced.assert_not_called()
