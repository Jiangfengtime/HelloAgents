# 真实大模型功能验证

本目录每个测试都使用 `.env` 中配置的真实大模型服务；没有预设回复、模拟客户端或 monkeypatch。工具执行的是本机实际 Python 实现。模型调用、参数解析、协议兼容、推理结果任一环节不符合断言，测试就会失败。

## 1. 配置与运行

在项目根目录的 `.env` 配置属于同一服务的以下三项：

```dotenv
LLM_MODEL_ID=你的实际模型ID
LLM_API_KEY=你的有效密钥
LLM_BASE_URL=你的模型服务兼容接口地址
```

不需要把密钥写入测试代码。此目录显式采用统一配置，不依赖其他服务商变量自动选择模型。使用既有 Python 3.10 环境：

```bash
# 先确认服务可用：只发一次模型请求，显示完整请求和响应
HELLO_AGENTS_LIVE_TEST=1 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/live/test_real_model.py::test_live_direct_model -v -s

# 全部真实场景；首个失败即停止，避免配置错误时重复请求
HELLO_AGENTS_LIVE_TEST=1 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/live -v -s -x

# 只验证原生工具调用
HELLO_AGENTS_LIVE_TEST=1 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/live/test_real_model.py::test_live_native_tools_lookup_then_calculate -v -s

# 只验证文本工具调用
HELLO_AGENTS_LIVE_TEST=1 /opt/anaconda3/envs/atguigu_ai/bin/python -m pytest tests/live/test_real_model.py::test_live_simple_text_tool -v -s
```

`-s` 展示实际发送的消息、模型原始 JSON 响应、工具执行日志和最终断言结果。默认不设置 `HELLO_AGENTS_LIVE_TEST=1` 时跳过本目录，普通回归测试不会自动消耗模型额度。一次完整运行预计约 20–30 次请求，实际次数受工具决策和停止条件影响；单次超时 60 秒，SDK 自动重试关闭。

PyCharm 可在 pytest 运行配置中选择上述解释器，把工作目录设为项目根目录，添加环境变量 `HELLO_AGENTS_LIVE_TEST=1`，并将附加参数设为 `-v -s -x`。

## 2. 九个真实场景

| 测试 | 真实执行与验证点 |
| --- | --- |
| `test_live_direct_model` | 发出真实请求计算137×29，断言结果3973及请求次数 |
| `test_live_multi_turn_memory` | 每次生成随机暗号，第一轮告知、第二轮回忆；验证回答和四条历史 |
| `test_live_stream` | 读取真实流式片段，验证内容及最终历史与片段拼接一致 |
| `test_live_simple_text_tool` | 模型输出文本工具标记，实际执行原计算器，将结果交回模型生成回答 |
| `test_live_native_tools_lookup_then_calculate` | 运行时随机单价，模型先调用实际本地查询工具，再调用原计算器算4件总价；验证两个工具均执行、结果正确、每个tool_call_id配对 |
| `test_live_react` | 真实模型生成Action，本地工具执行，Observation回填，再通过Finish结束 |
| `test_live_reflection` | 真实模型生成初稿并评审，按真实反馈选择停止或改写；检查轨迹和调用次数 |
| `test_live_plan_and_solve` | 真实模型生成恰好两步计划，执行器逐步调用真实模型，检查最终31及三次请求 |
| `test_live_memory_context_answer` | 真实进程内工作记忆写入随机代号、检索、构建上下文，再由真实模型依据证据回答 |

模型决策可能不稳定。这里不通过固定模型回复保证成功：不调用工具、没有符合ReAct/规划格式、遗漏证据或答错时，断言会直接指出问题。仅“模型回复声称调用了工具”不能通过工具执行测试。

## 3. 如何自己核对请求和结果

每次运行后，本机 `.local-test-results/live/` 会生成按用例名区分的 JSON。重复运行同一用例会覆盖此前记录。

记录包含测试结果状态、模型名称、服务主机、完整请求正文、完整非流式模型响应与流式文本片段；不记录 API key、请求头或整个 `.env`。此目录已加入 Git 忽略规则，避免推送运行记录。

重点查看：

1. `requests[].messages`：真正发送给服务的内容，是否包含历史、工具结果、检索证据。
2. `responses[].choices[].message`：模型实际返回的 content、tool_calls、arguments。
3. `role=tool` 与 `tool_call_id`：工具执行结果是否对应模型请求。
4. 终端的 `[实际计算器执行]` / `[实际价格查询]`：程序实际执行情况。
5. pytest 的 PASSED/FAILED 和报告的 outcome：是否满足行为断言。

HTTP事件仅观察真实客户端请求，不替换生成结果。原生工具Agent仍执行自身的底层客户端调用，其他Agent仍执行原有run方法。

## 4. 当前实测状态

2026-11-13，使用本机 `.env` 的模型 `deepseek-v4-flash` 和服务 `api.deepseek.com`，确实发送了一次真实非流式请求。

**服务返回401 authentication_error，提示API Key无效。** 因此基础测试实际失败；后续八个场景尚未完成真实服务验证。不是网络、工具解析或答案断言阶段的错误，也没有回退到模拟回答。

请在本地更新有效 `LLM_API_KEY`，确认模型ID与服务地址属于该账号可用配置，然后重新运行第一条命令。不要把密钥发到聊天或提交到Git。只有真实返回后断言通过，才能将相应场景记为已验证。

## 5. 功能范围

所有模型交互都是真实请求。价格查询和计算器是本机真实执行的工具，价格查询使用运行时生成的本地商品数据；工作记忆也是真实进程内组件。它们不需要外部商品系统或数据库。

本目录没有连接Qdrant、Neo4j、搜索API，也没有运行完整文档RAG入库、协议互操作或训练。`test_live_memory_context_answer` 验证的是工作记忆检索与上下文注入，不能将其解释为外部向量RAG已经验证。
