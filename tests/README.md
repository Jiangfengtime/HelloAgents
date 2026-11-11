# 核心功能测试与验证手册

这组例子帮助你在不调用付费模型、不启动向量数据库的情况下，验证 Agent 如何构造请求、执行工具、回填结果和维护状态。每个测试函数均可单独运行，也可在 PyCharm 中打断点观察输入输出。

## 1. 最快运行

在项目根目录执行。以下解释器已在本机验证可用（Python 3.10.0）：

```bash
# 只跑新增核心功能场景
HELLO_AGENTS_LIVE_TEST=0 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features -v

# 跑全部测试，明确关闭真实模型请求
HELLO_AGENTS_LIVE_TEST=0 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests -v

# 列出场景，不执行
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features --collect-only -q
```

也可将解释器替换为你自己配置好的 Python >= 3.10 环境。当前项目 `.venv` 是 Python 3.9.6，并缺少导入链要求的 huggingface_hub，尚不适合直接运行这些测试。本次没有修改它或安装新依赖。

依赖至少包含项目运行依赖、pytest，以及当前顶层导入链所需的 huggingface_hub。虽然评估属于扩展功能，当前包的顶层导入会触达该依赖；因此“不联网运行”仍需要预先准备本地依赖。

## 2. 场景覆盖地图

新增 **96 个参数化后的场景**，其中 93 个通过，3 个为严格标记的已知缺陷预期失败。加上原测试，全量结果：**100 passed、2 skipped、4 xfailed**；另有一条已有 Pydantic 弃用警告。此处是场景数量，不是源码行覆盖率，也不宣称穷尽所有输入。

| 测试文件 | 场景数 | 核心验证 |
| --- | ---: | --- |
| [test_agents.py](core_features/test_agents.py) | 25 | 多轮消息、清空历史、模型故障、工具禁用、多个文本调用、0/1/2 轮上限、未知/异常工具、流式写入、ReAct 结束与错误、反思重置、无效计划、ToolAware 嵌套 JSON 与回调 |
| [test_function_call.py](core_features/test_function_call.py) | 13 | 参数 schema、类型转换、多个 tool_call_id 精确配对、非法 JSON、工具上限强制 none、函数工具、工具故障回填、无工具对话 |
| [test_llm.py](core_features/test_llm.py) | 7 | 显式配置优先、环境读取、缺 key/地址、模型参数透传、客户端异常、流式空片段与惰性调用 |
| [test_memory_context.py](core_features/test_memory_context.py) | 12 | 工作记忆增删改查、用户与遗忘过滤、容量、TTL、Token 计数、管理器路由、空类型列表、上下文检索/筛选/结构/预算/最近历史/降级 |
| [test_rag_evaluation.py](core_features/test_rag_evaluation.py) | 18 | 分块重叠与来源、缺失/空/重复文件、嵌入批处理、入库 payload、写入失败、namespace 与阈值、空查询、检索故障、加权排序、入库到查询、GAIA 本地计分、RAG 问答证据与引用 |
| [test_tools.py](core_features/test_tools.py) | 21 | 运算优先级、括号、负数、数学函数、除零与非法表达式、注册覆盖/注销/清空、同名优先级、展开子工具、参数缺失、链式传值、模板缺失、异步结果配对、链错误中止缺陷 |

## 3. 按功能或单个场景运行

```bash
# 原生工具调用全过程
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features/test_function_call.py -v

# 只看多个工具调用的 ID 与结果怎样配对
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features/test_function_call.py::test_multiple_calls_match_results_by_id -v -s

# 查询迭代上限场景
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features -k iteration_limit -v

# 记忆与上下文
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features/test_memory_context.py -v

# RAG 与本地评估
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features/test_rag_evaluation.py -v

# 查看预期失败原因
/opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/core_features -rx -q
```

`-s` 显示被测框架的日志；`-v` 显示具体场景名；`-k` 按测试名筛选；`文件::函数名` 精确选择一个例子。PyCharm 里选择上述解释器、将工作目录设为仓库根目录，即可右键运行或调试测试函数。

## 4. 如何改成自己的例子

公共夹具在 [conftest.py](core_features/conftest.py)：

- `llm_factory([...])`：依次返回预设模型回复，保留 `requests` 中每次完整消息和参数。放入异常对象可模拟模型失败；用列表作为回复可模拟流式片段。
- `registry_and_tool`：提供真实注册表和可记录参数的本地工具。通过 `tool.calls` 验证工具是否真的执行、次数是否正确、类型转换是否生效。
- `response_factory` / `call_factory`：构造原生接口的完整响应，保留 tool_calls、arguments、ID，不把原生工具流程简化成普通文本回答。
- `tool_factory(fail=True)`：模拟工具抛异常，检查错误是否进入下一轮模型输入。
- `no_network`：新增测试执行期间阻断 socket 连接和域名解析，且记录尝试；即便被测代码吞掉连接异常，测试清理阶段也会失败。

例如，要检查“工具结果真的传回模型”，在 SimpleAgent 的第二次请求断言反馈内容；要检查“本轮过程是否污染跨轮历史”，检查 agent.get_history()；要检查“到上限是否继续执行工具”，同时断言 tool.calls 和模型请求次数。

示例可直接参考 `test_multiple_text_tool_calls_and_final_history`、`test_function_call_limit_forces_no_more_tools` 和 `test_rag_ask_injects_evidence_and_sources`。改回复、工具输入和期望输出即可扩展自己的用例。

## 5. 真实组件与替身的边界

| 实际执行 | 测试中替换 |
| --- | --- |
| Agent run 循环、消息拼装、历史保存、工具解析与回填 | 远程模型回复、接口客户端 |
| 注册表、计算器、装饰器展开、固定工具链、异步包装 | 会产生外部副作用的工具 |
| WorkingMemory 与只启用工作记忆的 MemoryManager | Qdrant、Neo4j 等长期后端未启动 |
| ContextBuilder 四阶段控制流 | tokenizer 使用一字符一计数的替身，未验证真实模型 Token 数 |
| RAG 分块、入库、过滤参数、排序、问答上下文与引用 | 文档格式转换、嵌入模型、向量存储返回值 |
| GAIA 单样本计分流程 | 使用自建少量本地样本，未下载官方数据 |

RAG 的 namespace 测试验证“过滤条件传递”和内存替身的筛选结果，不是对真实 Qdrant 的隔离或鉴权验证。异步批处理测试验证任务与结果配对，不宣称当前实现已经并发调度。

没有运行真实搜索、协议互操作、长期数据库、权重训练或官方 BFCL/GAIA 评测。这些需要专门的集成环境。RL 模块目前导入还会触达 datasets、transformers、trl；本次聚焦推理期核心路径，没有为跑本地测试而安装训练依赖。

## 6. 已知缺陷与预期失败

`XFAIL` 表示测试提出合理的行为要求，但当前实现存在已知缺陷；不能当作功能已通过。新增三项均设置 `strict=True`，修复后意外通过将导致测试套件失败，提醒同步移除标记。对确定抛异常的缺陷还限制了异常类型，避免无关错误被误认成预期失败。

| 测试 | 当前问题 | 预期行为 |
| --- | --- | --- |
| `test_simple_agent_remove_tool_contract` | 调用了注册表不存在的 unregister_tool | 从注册表移除工具并正确返回结果 |
| `test_manager_empty_types_should_return_no_results` | 空类型列表计算配额时除零 | 返回空列表 |
| `test_chain_should_stop_after_failed_tool` | 未找到工具返回错误字符串，链仍继续执行后续动作 | 失败后停止后续动作 |
| 原有 `test_reflection_negation_should_not_stop` | “不能认为无需改进”命中停止短语 | 正确识别否定反馈并继续改写 |

本次只补测试与使用说明，没有把功能修复混入测试提交。后续修复时，可先只运行对应测试，用 `--runxfail` 显示真实失败，再修改实现验证。

## 7. 可选真实模型验证

原有 [test_live_llm.py](test_live_llm.py) 默认跳过。显式开启时才会使用本机 `.env` 中的真实服务配置，并可能产生调用费用：

```bash
HELLO_AGENTS_LIVE_TEST=1 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/test_live_llm.py -v -s
```

这两项只验证真实模型响应和 SimpleAgent 基础链路，不代表所有模型工具调用、数据库与协议功能都已实测。你在该文件中的原有本地改动已保留，不属于这次提交。
