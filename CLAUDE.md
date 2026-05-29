# AGENT RUNTIME 项目指南 (Project Brief)

## 1.全局语言偏好 (Global Language Preference)

**核心指令**：在整个对话和代码生成过程中，你必须**始终使用简体中文**与用户交流。

1.  **回答语言**：所有对用户的回复、解释、说明必须使用简体中文。
2.  **代码注释**：所有生成的代码注释必须使用简体中文。
3.  **文档字符串**：所有 Docstrings 必须使用简体中文。
4.  **终端输出说明**：虽然终端（Bash）本身的报错可能是英文，但你在解释“刚才发生了什么”时必须用中文翻译。
5.  **思考过程**:在进行复杂的逻辑推理、步骤规划或问题拆解（即生成 <thinking> 或类似内部思考块）时，请强制使用简体中文进行思维链推演。不要使用英文思考后再翻译，而是直接用中文构建你的逻辑框架，以便用户能实时理解你的决策路径和排查思路。


## 2. 项目目的 (Purpose)
本项目并非开发“聊天机器人”或“普通学习助手”，而是构建一个**可治理的单 Agent Runtime 实验平台**，用于**系统性训练和理解 Agent Engineering 的核心能力**。  

项目围绕 **“Agent Engineering 学习助手”** 场景展开，通过学习路径规划、知识检索、任务执行、长期记忆、状态流转与学习反馈等流程，**真实实现并验证**以下关键模块：  
- State 管理（状态机驱动）  
- Edge 路由（条件流转）  
- Tool 调度（工具链执行）  
- Memory 生命周期（短期/长期/知识记忆）  
- RAG 检索链路（检索增强生成）  
- 异常恢复（Recovery）  
- 效果评估（Eval）  
- Reflection 反思机制  

**核心目标**：深入理解现代 Agent 系统“为什么需要 Graph（状态图）”、“为什么需要状态机”、“为什么 Tool 与 Memory 管理比 Prompt 更重要”，以及一个 Agent Runtime 在真实运行中如何被**治理、演化、观测与迭代**。  


## 3. 预期实现效果 (Expected Outcome)
最终系统将实现一个具备 **“学习规划 + 知识检索 + 长期记忆 + 自我反思 + 故障恢复”** 的单 Agent 学习执行体，具体特征如下：  

### 3.1 核心能力（对应架构图模块）
| 模块          | 子能力/组件                                                                 | 实现目标                                                                 |
|---------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| **交互层**    | Streamlit Web UI、FastAPI 接口                                              | 支持用户通过 Web 或 API 与 Agent 交互                                     |
| **State 流转** | LangGraph 状态机（PLANNER→DECIDE→RETRIEVE→EXECUTE→REFLECT→END）、状态管理/边路由/中断恢复/并发控制/错误处理 | 实现”规划→决策→检索→执行→反思→结束”的闭环状态流转。错误处理原则：任何节点出错立即进入 REFLECT，节点不做本地重试；REFLECT 是唯一的重试决策点，通过 next_action(retry/end)+retry_target_node(PLANNER/RETRIEVE/EXECUTE) 决定去向；retry_count=REFLECT 执行次数，达到 max_retries 后强制终止 |
| **Tool 链路**  | 工具注册（search_docs/memory_write/memory_read/context_store/fallback）、工具执行流程（选择→校验→执行→返回） | Runtime 原子能力层：知识检索、记忆读写、状态持久化、统一兜底              |
| **RAG 检索增强** | Query Rewrite（改写）、向量检索（Vector Search）、Rerank（重排序）、Context Assemble（上下文组装）、文档加载/切分/嵌入 | 基于知识库（PDF/MD/HTML）实现精准检索增强，支持多轮对话上下文感知         |
| **Memory 记忆系统** | 短期记忆（Redis/In-Memory，会话上下文）、长期记忆（SQLite/Redis/JSON，用户画像/进度/偏好）、知识记忆（向量库/KV，事实/经验/反思） | 实现“会话级→用户级→知识级”的记忆生命周期管理，支持记忆更新与检索         |
| **反思机制**  | 自我评估（置信度）、错误分析（原因归因）、改进建议（下一步行动）、记忆更新（写入长期记忆） | 支持 Agent 对执行结果自我反思，优化后续决策                               |
| **异常处理&Recovery** | 重试（Retry）、降级（Fallback）、兜底回复（Guardrail）、人工干预（可选）       | 工具/检索/模型异常时自动恢复，保证服务稳定性                             |
| **评估系统 (Eval)** | RAG 评估（命中率/覆盖率）、回答质量评估（LLM-as-Judge）、任务成功率、幻觉率评估 | 量化 Agent 执行效果，支持迭代优化                                         |
| **可观测性**  | LangSmith Tracing、日志系统、监控&告警                                       | 追踪 Agent 每一步状态流转、Tool 调用、检索结果、置信度与恢复过程           |


### 3.2 场景化能力（学习助手）
用户可围绕 **Agent、RAG、LangGraph、Prompt、Memory** 等领域进行学习：  
- Agent 基于知识库（RAG）检索增强，动态生成**个性化学习路线**；  
- 记录用户**学习状态与历史进度**，在多轮对话中维护**长期记忆**；  
- Tool 调用失败、检索失败或模型输出异常时，自动触发**恢复与兜底**；  
- 系统具备**可观测性与评测能力**，可追踪 Agent 全流程状态，验证工程化效果。  


## 4. 目录结构与模块职责（参考架构图）
AGENT RUNTIME/
├── main.py # 入口文件
├── requirements.txt # 项目依赖
├── .vscode/ # VS Code 项目配置
├── runtime/ # 状态机、边路由、恢复等核心逻辑
│   ├── node_planner.py      # PLANNER 节点
│   ├── node_decide.py       # DECIDE 节点
│   ├── node_retrieve.py     # RETRIEVE 节点
│   ├── node_execute.py      # EXECUTE 节点
│   ├── node_reflect.py      # REFLECT 节点
│   ├── state_graph.py # 状态图定义（LangGraph）
│   ├── state_manager.py # 状态管理（AgentState Schema）
│   ├── edge_router.py # 边路由（条件流转）
│   ├── checkpointer.py # 中断恢复（Checkpointer）
│   └── recovery_handler.py # 异常处理与恢复
├── tools/ # Runtime 工具层（原子能力）
│ ├── search_docs.py # 知识检索工具
│ ├── memory_write.py # 记忆写入工具
│ ├── memory_read.py # 记忆读取工具
│ ├── context_store.py # 会话状态持久化
│ ├── fallback.py # 统一兜底工具
│ ├── tool_registry.py # 工具注册与发现
│ └── tool_executor.py # 工具执行流程（选择→校验→执行→返回）
├── memory/ # 三种记忆系统
│ ├── short_term_memory.py # 短期记忆（会话上下文）
│ ├── long_term_memory.py # 长期记忆（用户画像/进度）
│ ├── knowledge_memory.py # 知识记忆（事实/经验）
│ ├── vector_store.py # 向量库操作（FAISS/Chroma）
│ └── memory_lifecycle.py # 记忆生命周期管理
├── rag/ # RAG 检索增强全流程
│ ├── query_rewrite.py # 查询改写
│ ├── vector_retriever.py # 向量检索
│ ├── rerank.py # 重排序
│ ├── context_assembler.py # 上下文组装
│ ├── document_loader.py # 文档加载（PDF/MD/HTML）
│ ├── text_splitter.py # 文本切分
│ └── embedding.py # 文本嵌入（Embedding）
├── reflection/ # 反思机制与评估系统
│ ├── self_reflection.py # 自我评估（置信度）
│ ├── error_analysis.py # 错误分析（原因归因）
│ ├── improvement_suggestion.py # 改进建议（下一步行动）
│ ├── eval_metrics.py # 评估指标（RAG/回答质量/任务成功/幻觉率）
│ └── hallucination_detector.py # 幻觉检测
├── interface/ # 用户交互层
│ ├── streamlit_app.py # Streamlit Web UI
│ ├── fastapi_app.py # FastAPI 接口
│ └── api_routes.py # API 路由定义
├── observability/ # 可观测性与日志
│ ├── tracer.py # LangSmith 追踪
│ ├── logger.py # 日志系统
│ └── monitoring.py # 监控与告警
├── engine/ # 系统运转的核心动力(配置、Prompt、模型调用、安全)
│ ├── config.py # 配置管理
│ ├── prompt_manager.py # Prompt 模板管理
│ ├── model_manager.py # 模型管理（LLM 调用）
│ └── security.py # 安全与权限
└── tests/ # 测试文件
├── test_state_graph.py # 状态图测试
└── test_tool_chain.py # 工具链测试
## 5. 开发与迭代原则
1. **模块化设计**：严格遵循目录结构，每个模块职责单一（如 `runtime` 只处理状态机，`tools` 只处理工具逻辑）。  
2. **可观测性优先**：所有核心流程（状态流转、Tool 调用、RAG 检索、记忆更新）必须支持追踪与日志。  
3. **异常即数据**：异常处理（Recovery）不仅是“兜底”，更要记录错误类型、上下文，用于后续反思与优化。  
4. **迭代式验证**：每完成一个模块（如 State 流转、RAG 检索），需通过测试或示例验证其”可治理性”（如状态是否闭环、检索是否精准）。  

## 5.1 核心设计决策（后续开发必须遵循）
- **5 节点状态机**：PLANNER → DECIDE → RETRIEVE → EXECUTE → REFLECT（→ END 或回退任一节点）
- **错误处理**：任何节点出错立即进 REFLECT，节点不做本地重试
- **retry_count**：REFLECT 执行次数（由 reflect_node 自增），达到 max_retries 后即使 REFLECT 建议 retry 也强制 END
- **reflection.next_action**：仅 `”end”` 和 `”retry”` 两种；`”retry”` 必须附带 `retry_target_node` ∈ {PLANNER, RETRIEVE, EXECUTE}
- **AgentState**：共 27 个字段（含 6 个顶层治理字段），完整定义见 `runtime/state_manager.py`
- **LLM 驱动节点**：PLANNER / DECIDE / REFLECT 由 LLM 产出核心数据；LLM 失败时各节点有模板降级兜底
- **意图分流**：PLANNER 产出 `mode`（learn / qa），learn 走完整 5 节点，qa 走"检索→合成→评估"；工具选择由 DECIDE 负责
- **答案合成在 EXECUTE**：QA 意图下，EXECUTE 基于 retrieved_context 做受约束答案合成；REFLECT 只评估 groundedness + completeness，不判正确性
- **learn 模式 REFLECT 不评估检索质量**：学习计划由 PLANNER 的 LLM 直接生成，不依赖检索结果；因此 learn 模式的 REFLECT 跳过 retrieval_score/retrieval_attempted 信号，仅评估工具执行状态和计划质量；LLM 或规则引擎若建议 retry RETRIEVE，强制改为 retry PLANNER
- **治理字段**：AgentState 顶层含 mode / retrieval_score / groundedness_score / completeness_score / answer_source / retry_reason / fallback_triggered / fallback_reason / retry_count / error，共 10 个字段

## Python 环境与导入规范
本项目使用根目录 `.venv` 虚拟环境。执行 Python 或 Pip 时，必须调用 `.venv/bin/python` 和 `.venv/bin/pip`（Linux/Mac）或 `.venv\Scripts\python.exe`（Windows），禁止直接调用 `python` 或 `pip`。

项目根目录（`AGENT_RUNTIME/`）为 Python 包搜索起点（见 `.vscode/settings.json`）。`runtime/` 和 `tools/` 等均为 Python 包（含 `__init__.py`），跨包导入使用标准包路径，如 `from runtime.state_manager import AgentState`。

## 6. 开发执行协议 (必须严格遵守)
### 自检清单 (每步必做)
- [ ] 文件存在性检查。
- [ ] 语法检查 (`python -m py_compile`)。
- [ ] 导入检查 (无 ModuleNotFoundError)。
- [ ] 架构一致性检查 (未偏离上述目录结构)。
- [ ] 每次开发完成解释对应修改点 (描述旧版本与新版本的区别)。


### 测试规范
- **临时调试**：可直接通过 Bash 工具 + `python -c "..."` 内联运行快速验证，不必落盘。
- **重要验证**（模块验收、链路回归、逻辑正确性）：必须将测试逻辑写入 `tests/` 目录下的 `.py` 脚本，文件命名清晰反映测试目标（如 `test_decide_matching.py`）。用户需能阅读测试逻辑和运行结果，以便判断测试是否充分。
- 每个 Phase 结束时，确保 `tests/` 下的 pytest 用例全部 PASSED。

### 编码约束
- 写代码前必须回顾本文件中的”项目目的”和”目录结构”。
- 涉及核心模块 (runtime/rag/memory) 必须写中文 Docstring。
- 严禁绕过架构设计 (如跳过 State 直接调用 LLM)。
- 遇无法解决的报错立即停止，输出日志，不假装成功。

### Karpathy 编码指南 (andrej-karpathy-skills:karpathy-guidelines)
每次编写或修改代码时必须遵循以下 4 条原则：
1. **想清楚再写**：明确假设、不确定就问、有更简单的方案就说
2. **简单优先**：最少代码解决问题，不写”以后可能用到”的抽象，200 行能干的不用 400 行
3. **精准修改**：只改任务相关的代码，不顺手重构、不改格式、不删不相干的死代码；自己引入的孤儿导入/变量必须清
4. **目标驱动**：每步定义验收条件，写完立刻验证，循环直到通过

### 外部建议评估原则
接收到 GPT 或其他外部建议时，必须逐条批判性评估，不得无脑认同：
1. **区分方向 vs 方案**：建议的"方向"可能对，但"实现方案"可能过度。方向认同不代表方案采纳——找更简单的手段达成同一目标
2. **算量级**：建议提的优化（向量搜索、淘汰策略、加权公式）是否在你当前数据规模上有意义？几十条数据谈淘汰策略就是浪费
3. **砍掉"以后可能用到"**：如果建议解决的问题当前不存在或规模不到，直接砍掉。真需要时再加
4. **逆向思维**：先问"不做它会有什么实际损失？"——如果答案是"暂时没有"，就不做

关键判断链条：这个建议要解决什么问题 → 这个问题在当前项目中真实存在吗 → 最简单的解法是什么 → 有没有更简单的手段

### 开发交付规范
每完成一个开发步骤（Step），必须输出一张 **修改汇总表**，让用户一目了然本次改了什么：

| 文件 | 修改内容 | 修改效果 | 修改理由 |
|---|---|---|---|
| `path/to/file.py` | 一句话描述做了什么改动 | 改后产生的行为变化 | 为什么要这样改 |

表格要求：
- 行数 = 受影响的文件数，每个文件一行
- 内容简洁，不用展开代码细节
- “修改效果”要描述可观测的行为变化（如 “LLM 失败时输出降级日志”）
- “修改理由”要说明设计意图（如 “遵循 node_start/node_end 契约”），不是重复修改内容

## 7.依赖安装
`requirements.txt` 当前包含以下核心依赖：
- **langgraph** — 状态机框架
- **fastapi** / **uvicorn** — API 服务
- **pydantic** — 数据校验（工具 Schema）
- **redis** — 短期记忆存储
- **faiss-cpu** — 向量检索
- **streamlit** — Web UI
- **python-dotenv** — 环境变量管理
- **langchain-openai** / **langchain** — LLM 调用
- **pytest** — 测试框架
- **tenacity** — 超时/重试

安装命令：
```bash
.venv/Scripts/pip install -r requirements.txt
```

## 8. 全项目实现总览（Phase 13 完成后）

### 8.1 全局架构

```
用户输入 → PLANNER(意图分流+水平检测) → DECIDE(工具选择) → RETRIEVE(Query Rewrite + FAISS) → EXECUTE(工具执行/QA合成) → REFLECT(规则参谋→LLM司令→规则预备) → END
              │         ↑                                                                              │
              │         └─────────────── retry (PLANNER / RETRIEVE / EXECUTE) ──────────────────────────┘
              │
              └── 未说明水平 → 反问用户 → END（下次带水平信息再来，PLANNER 读取长期记忆生成差异化计划）
```

降级策略：每个 LLM 驱动节点（PLANNER/DECIDE/REFLECT）有独立的降级路径，LLM 失败时自动切换。REFLECT 降级使用规则引擎结果而非硬编码 end。

AgentState 治理字段（10 个）：`mode`, `retrieval_score`, `groundedness_score`, `completeness_score`, `answer_source`, `retry_reason`, `fallback_triggered`, `fallback_reason`, `retry_count`, `error`

### 8.2 文件功能总览

| 目录 | 文件 | 功能 | 实现方式 |
|---|---|---|---|
| **engine/** | `config.py` | 配置管理：读取 .env 统一暴露 LLM 和 Embedding 两套配置（base_url/api_key 可分开） | `os.getenv()` + 模块级缓存；`get_embedding_config()` 独立读取 embedding 端点 |
| | `model_manager.py` | LLM 调用封装：文本生成 + Pydantic 结构化输出 | `ChatOpenAI` (LangChain)，temperature=0.3，含异常捕获 + logger 记录耗时 |
| | `prompt_manager.py` | 6 个 Prompt 模板定义 + `render()` 变量替换 + 工具列表格式化 | 字符串常量 + `str.format()` |
| | `security.py` | 空壳 | — |
| **runtime/** | `state_manager.py` | AgentState TypedDict（27+ 字段）+ `create_initial_state(user_input, history)` | TypedDict + 显式初始化所有字段 |
| | `state_graph.py` | LangGraph StateGraph 组装：5 节点注册 + 5 条件边 + 编译 | `build_graph()` + `run_graph(user_input, session_id, history, user_id)` |
| | `edge_router.py` | 5 个条件路由函数 + Route 常量（含 `router_after_planner`：反问水平→END） | 纯函数，输入 state → 返回下一节点 |
| | `node_planner.py` | PLANNER：关键词预判+水平检测+已完成主题查重→ LLM 生成计划；未说明水平时反问用户 | `_detect_level()` + `_plan_via_llm()` + `_plan_fallback()` |
| | `node_decide.py` | DECIDE：LLM 选工具+生成参数；LLM 失败走关键词规则匹配 | `_decide_via_llm()` + `_decide_via_keyword()` + `_parse_json()` |
| | `node_retrieve.py` | RETRIEVE：Query Rewrite → FAISS 检索 → 产出 retrieved_context | `rewrite()` + `VectorRetriever.retrieve()` |
| | `node_execute.py` | EXECUTE：learn 模式执行工具调用；qa 模式 LLM 合成答案；降级取首条文档 | `_execute_tool()` + `_execute_qa()` + `_synthesize_via_llm()` |
| | `node_reflect.py` | REFLECT：规则引擎(3 模块)→LLM override→规则降级；learn 满意后标记主题完成 | error_analysis + improvement_suggestion + self_reflection + mark_topic_completed |
| | `checkpointer.py` | 空壳 | LangGraph MemorySaver 已覆盖 |
| | `recovery_handler.py` | 空壳 | v3 路由"任何错误→REFLECT"已覆盖 |
| **tools/** | `tool_registry.py` | `ToolRegistry` + `ToolInfo`：显式注册 2 个 Runtime 工具（search_docs + memory_write） | Pydantic `BaseModel` 参数 Schema |
| | `tool_executor.py` | `ToolExecutor`：校验→超时(tenacity)→执行→标准化 `ToolOutput` | Pydantic 校验 + `tenacity.retry` |
| | `search_docs.py` | 知识检索工具（已被真实 RAG 替代，保留用于 learn 模式） | 关键词匹配文档库 |
| | `memory_write.py` | `MemoryWriteInput` Schema + `run()`：写入 SQLite 长期记忆 | Pydantic + `long_term_memory.save()` |
| | `memory_read.py` | `MemoryReadInput` Schema + `run()`：按 key/category 读取长期记忆 | `long_term_memory.load()` |
| | `context_store.py` | 空壳 | — |
| | `fallback.py` | 统一兜底工具：返回安全默认回复 | 硬编码安全消息 |
| **memory/** | `long_term_memory.py` | SQLite 长期记忆：`save()` / `load()` / `load_recent_experiences()` / `load_experience_summaries()` + 用户画像 `get_user_profile()` / `mark_topic_completed()` | `sqlite3` 单表 memories(key/value/category/timestamp/session_id) |
| | `short_term_memory.py` | 空壳 | 短期记忆由 AgentState.short_term_memory["buffer"] 承载 |
| | `knowledge_memory.py` | 空壳 | 经验数据量级不到触发 |
| | `vector_store.py` | 空壳 | FAISS 实现在 `rag/vector_store.py` |
| | `memory_lifecycle.py` | 空壳 | 经验数据量级不到触发 |
| **rag/** | `document_loader.py` | 加载 data/knowledge/ 下 12 个 MD 文件 + YAML 元数据解析 | `os.listdir` + `str.partition` 按中/英文冒号切分键值对 |
| | `text_splitter.py` | 按 `##` 标题语义切分 chunk，chunk_size=500, overlap=50 | 递归按标题层级切分，确保标题完整性 |
| | `embedding.py` | `EmbeddingProvider` ABC + `QwenEmbeddingProvider`（千问 text-embedding-v3，独立 endpoint） | HTTP POST + 3 次重试；通过 `get_embedding_config()` 走独立 base_url |
| | `vector_store.py` | `FAISSVectorStore`：建索引(batch_size=10)、搜索、持久化 save/load | FAISS `IndexFlatIP` + 内存映射 |
| | `vector_retriever.py` | `VectorRetriever`：统一检索入口，含自动建索引 + 模块级单例 | embedding → FAISS search → 返回 top-K |
| | `query_rewrite.py` | LLM Query Rewrite：改写含指代词的查询为完整独立查询 | `_needs_rewrite()` 快速跳过 + LLM 改写 + 失败回退原 query |
| | `rerank.py` | 空壳 | top-K 质量问题不严重 |
| | `context_assembler.py` | 空壳 | — |
| **reflection/** | `error_analysis.py` | 错误分类器：6 种 error_type + recoverable + severity（纯函数） | 5 级优先级判断：max_retries > fallback > error > tool > retrieval |
| | `improvement_suggestion.py` | 建议生成器：6 种 reason_code + next_action/retry_target_node（纯函数） | 不可恢复→end；可恢复按类型路由；质量信号独立判断 |
| | `self_reflection.py` | 置信度评估：4 因子加权 `g×0.4 + c×0.3 - fb×0.2 - hl×0.3`（纯函数） | 缺失值默认 0.5，结果 clamp 到 [0,1] |
| | `hallucination_detector.py` | 幻觉检测：3 条规则（接地低+RAG/空上下文/长回答低接地） | 纯规则驱动，阈值可配 |
| | `eval_metrics.py` | 5 类确定性指标：RAG/工具/回答/流程/治理 | 纯函数，输入最终 AgentState → 输出结构化指标 |
| **observability/** | `logger.py` | `NodeLogger`：node_start/node_end/node_error 契约，JSON 输出到 stdout | `time.time()` 计时 + 治理字段提取 |
| | `tracer.py` | `TraceCollector` + `trace_node` 包装器 + `reset_tracer` | 记录 node/duration/fields_produced/state_diff/tool_calls |
| | `monitoring.py` | 空壳 | logger + tracer 已覆盖 |
| **interface/** | `streamlit_app.py` | 治理驾驶舱：输入→回答+节点时序+治理仪表+历史经验+多轮对话 | `st.session_state.messages` 累积历史，最近 5 轮传入 run_graph |
| | `fastapi_app.py` | FastAPI 应用入口：`POST /chat` 单端点 | 纯透传，零业务逻辑 |
| | `api_routes.py` | `ChatRequest` 模型 + `handle_chat()` 处理函数 | 调用 run_graph() + 组装 agent_state/metrics/hallucination/trace |
| **eval/** | `eval_runner.py` | 批量评估：读 eval_questions.json → 逐题 run_graph → 输出 results.json | stdout 重定向静默 logger/tracer 噪音 |
| | `eval_reporter.py` | 聚合报告 + A/B 对比：成功率/分布(P50/P95)/分组/--compare | 纯 Python 统计，无外部依赖 |
| **tests/** | `test_state_graph.py` | 8 个状态图流程测试（happy path/错误路由/retry/上限） | pytest |
| | `test_tool_chain.py` | 27 个工具链测试（注册/执行/DECIDE/EXECUTE） | pytest |
| | `test_model_manager.py` | 41 个测试（配置/Prompt/LLM/JSON解析/REFLECT） | pytest，标记 `@pytest.mark.integration` 跳过 LLM 敏感用例 |
| | `test_memory.py` | 15 个长期记忆测试 | pytest |
| | `test_rag.py` | 17 个 RAG 管道测试 | pytest，标记 `@pytest.mark.integration` |
| | `test_observability.py` | 42 个测试（Logger/Tracer/Metrics/Hallucination） | pytest |
| | `test_reflection.py` | 44 个测试（error/suggestion/confidence/experience 闭环） | pytest |
| | `test_llm_failure.py` | 7 个 LLM 异常压测（PLANNER/DECIDE/REFLECT 降级 + 全面崩溃恢复） | pytest + `unittest.mock.patch` |

### 8.3 空壳文件清单（不填的原因）

| 文件 | 不做原因 |
|---|---|
| `runtime/checkpointer.py` | LangGraph MemorySaver 已覆盖 |
| `runtime/recovery_handler.py` | v3 路由"任何错误→REFLECT"已覆盖 |
| `memory/short_term_memory.py` | AgentState.short_term_memory["buffer"] 字段已够用 |
| `memory/knowledge_memory.py` | 经验量级（几十条）不到需要向量库的程度 |
| `memory/vector_store.py` | FAISS 实现在 rag/vector_store.py |
| `memory/memory_lifecycle.py` | 几十条数据谈淘汰策略是过度设计 |
| `rag/rerank.py` | 12 篇文档、96 个 chunk，top-5 质量问题不严重 |
| `rag/context_assembler.py` | 当前组装逻辑在 node_execute._format_context() 中 |
| `observability/monitoring.py` | logger + tracer 已覆盖 |
| `engine/security.py` | 实验平台不需要 |
| `tools/context_store.py` | 场景不明确 |