# Agent Runtime — 可治理单智能体实验平台

**一个用于系统性理解和训练 Agent Engineering 核心能力的实验平台。不是聊天机器人，是一个可观测、可评估、可对比的 Runtime 治理系统。**

### 亮点

- **基于用户画像的自适应学习路径**：长期记忆（SQLite）记录每个用户的学习进度和已完成主题，PLANNER 根据用户水平（入门/中等/进阶）生成差异化学习计划，已学主题自动避重
- **三层反思决策链**：规则引擎（纯函数）预分析 → LLM 综合判断（可覆盖规则） → 规则引擎降级兜底，REFLECT 是整个系统唯一的重试决策点
- **LLM 全链路降级容错**：每个 LLM 驱动节点都有独立降级路径，全面崩溃仍返回有效状态，已通过 7 项异常压测验证
- **确定性评估与 A/B 对比**：12 题 Golden Dataset + 批量运行器 + 聚合报告（P50/P95/分组分析），支持双报告对比验证改动效果

---

## 架构总览

```
用户输入 → PLANNER(意图分流+水平检测) → DECIDE(工具选择) → RETRIEVE(Query Rewrite + FAISS) → EXECUTE(执行/合成) → REFLECT(规则参谋→LLM司令→规则预备) → END
              │         ↑                                                                              │
              │         └───────────────── retry (PLANNER / RETRIEVE / EXECUTE) ────────────────────────┘
              │
              └── 未说明水平 → 反问用户 → END（等用户回复水平后下一轮再生成计划）
```

### 5 节点状态机

| 节点 | 职责 | 驱动方式 |
|---|---|---|
| **PLANNER** | 意图分类（learn/qa）+ 水平检测 + 学习计划生成；未说明水平时反问用户 | LLM + 关键词预判 + 长期记忆画像 |
| **DECIDE** | 工具选择 + 参数生成 | LLM，失败降级关键词规则 |
| **RETRIEVE** | Query Rewrite + FAISS 向量检索 + 质量阈值过滤 | LLM 改写 + Embedding 检索 |
| **EXECUTE** | 工具执行 / QA 受约束答案合成 | LLM 合成，失败降级首条文档 |
| **REFLECT** | 规则引擎预分析 → LLM 综合判断 → 规则引擎降级；learn 模式标记学习进度 | 三层决策链 + 长期记忆回写 |

### 核心设计决策

- **REFLECT 是唯一重试决策点**：任何节点出错不进本地重试，统一进 REFLECT 判断 retry/end，`retry_count` 由 REFLECT 自增
- **LLM 双轨降级**：PLANNER/DECIDE/REFLECT 每个 LLM 调用都有独立的确定性降级路径，全面崩溃仍可返回有效状态
- **检索质量阈值 0.45**：FAISS 内积 < 0.45 自动清空检索结果，避免对领域外问题的无效重试
- **learn 模式不评估检索质量**：学习计划由 LLM 直接生成，REFLECT 跳过检索分数评估，retry 不建议 RETRIEVE
- **治理字段透传**：10 个治理信号（groundedness/completeness/fallback/hallucination 等）贯穿全部节点
- **规则优先**：`error_analysis` + `improvement_suggestion` + `self_reflection` 三个纯函数模块在 LLM 调用前产出结构化建议

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 配置 LLM（Ollama 本地） + Embedding（千问）
# .env 内容：
#   OPENAI_API_KEY=ollama
#   OPENAI_BASE_URL=http://localhost:11434/v1
#   MODEL_NAME=qwen2.5:latest
#   EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
#   EMBEDDING_API_KEY=sk-xxx  # 千问 API Key

# 构建 FAISS 索引（首次运行自动触发）
python -c "from rag.vector_retriever import get_retriever; get_retriever()"

# 启动治理驾驶舱
streamlit run interface/streamlit_app.py

# 或启动 API
uvicorn interface.fastapi_app:app --port 8000
```

---

## 项目结构

```
├── runtime/          # 状态机核心：5 节点 + 状态图 + 边路由
├── rag/              # RAG 检索：文档加载→切分→Embedding→FAISS→Query Rewrite
├── reflection/       # 反思治理：错误分类/建议生成/置信度/幻觉检测/指标
├── memory/           # 记忆系统：SQLite 长期记忆 + 经验检索
├── tools/            # 工具层：5 个 Runtime 原子工具
├── observability/    # 可观测性：结构化日志 + Trace + 指标
├── engine/           # 引擎：配置/Prompt/LLM 调用
├── interface/        # 交互：Streamlit 驾驶舱 + FastAPI
├── eval/             # 评估：批量运行 + 聚合报告 + A/B 对比
├── tests/            # 201 个测试用例
├── data/knowledge/   # 12 篇 MD 知识库（Agent/RAG/LangGraph 等）
└── data/vector_store/ # FAISS 索引持久化
```

---

## 实验能力

### 批量评估
```bash
python eval/eval_runner.py             # 12 题 golden dataset → results.json
python eval/eval_reporter.py           # 聚合报告（P50/P95/分组分析）
python eval/eval_reporter.py --compare results_a.json results_b.json  # A/B 对比
```

### A/B 对比示例：LLM 模式 vs 关键词模式

| 指标 | keyword | llm | 差异 |
|---|---|---|---|
| 成功率 | 100% | 100% | - |
| 平均耗时 | 477ms | 7671ms | **16x** |
| 平均检索分 | 0.426 | 0.426 | 相同（检索确定性） |

### 异常压测
```bash
pytest tests/test_llm_failure.py -v   # 7 个故障场景：超时/脏JSON/空响应/全面崩溃
```

---

## 技术栈

- **State Machine**: LangGraph + MemorySaver
- **RAG Pipeline**: FAISS + Query Rewrite + Semantic Chunking + 检索质量阈值
- **Prompt**: 7 套 Prompt 模板 + 变量渲染 + 工具列表格式化
- **Eval**: 12 题 Golden Dataset + 批量运行器 + 聚合报告 + A/B 对比
- **Observability**: 结构化 JSON 日志（node_start/end/error 契约）+ TraceCollector + 确定性格标 + 幻觉检测
- **Memory**: SQLite 长期记忆 + 对话堆栈短期记忆 + 经验检索
- **UI**: Streamlit 治理驾驶舱 + FastAPI 单端点
- **测试**: pytest（201 用例）+ LLM 异常压测

---

## 设计说明

### 为什么 REFLECT 在 LLM 失败时用规则引擎结果而不是硬编码 end？

规则引擎（`error_analysis` + `improvement_suggestion`）已经能基于实际错误信号判断 retry/end。如果 LLM 不可用就直接 end，会丢失可恢复错误的补救机会。规则引擎是"没 LLM 时能用的最好替代"，不是"最安全的兜底"。

### 为什么用内存积 0.45 作为检索阈值？

实测数据：相关查询 top-1 分数 ≥ 0.48，不相关查询 ≤ 0.45。0.45 正好落在中间，不误杀也不漏网。低于此阈值的查询在知识库中无对应内容，清空结果避免 3 轮无效 retry。

### 为什么不用 FAISS 做经验记忆？

经验数据几十条量级，SQL 的 `WHERE error_type=? ORDER BY timestamp DESC` 比向量搜索更快更准。向量检索只在数据量上千且有语义相似需求时才值得。
