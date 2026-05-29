"""
Prompt 模板管理
定义各节点的提示词模板，提供变量替换接口
"""
import json

# ============================================================
# DECIDE 节点模板
# ============================================================

DECIDE_TOOL_SELECTION = """你是一个 Agent 系统的决策模块。

根据用户输入（含对话历史），从可用工具列表中选择最合适的工具。

【对话历史】
{conversation_history}

【用户输入】
{user_input}

【可用工具列表】
{tools_list}

【要求】
1. 选择最能满足用户意图的工具
2. 如果用户意图模糊，默认选择 search_docs（检索知识库）
3. confidence 表示你对选择的把握（0.0 完全不确信 ~ 1.0 非常确信）
4. 工具名必须是上述列表中的某个 name 值

【输出格式】只输出 JSON，不要其他任何内容：
{{"tool_name": "选择的工具名", "reason": "选择理由", "confidence": 0.0}}"""

DECIDE_ARGUMENTS = """你是一个 Agent 系统的参数生成模块。

根据用户输入和选定工具的参数定义，生成调用该工具所需的参数。

【用户输入】
{user_input}

【选定工具】
{name}

【工具描述】
{description}

【参数定义】
{parameters_schema}

【要求】
1. 参数值必须符合参数定义中的类型
2. 必填字段（required=true）必须赋值
3. 从用户输入中提取相关信息填入参数中，没有对应信息则使用合理默认值

【输出格式】只输出 JSON 对象，不要其他任何内容：
{{"字段名": "字段值", ...}}"""

# ============================================================
# PLANNER 节点模板
# ============================================================

PLANNER_PLAN = """你是一个 Agent 系统的规划模块。

首先判断用户意图，然后根据模式生成对应的计划。

【对话历史】
{conversation_history}

【用户输入】
{user_input}

【用户水平】
{user_level}

【已完成主题（避免重复推荐）】
{completed_topics}

【学习目标】
{goals}

【模式判断】
- "learn"：用户想系统学习某个主题，需要学习路径规划
- "qa"：用户只是想问一个问题，快速得到答案

判断标准：如果用户输入是简短的问题（如"什么是X"、"X有什么用"、"介绍一下X"、"X是什么"），选 qa；
如果用户输入是学习请求（如"学习X"、"帮我规划X的学习"、"我要学X"），选 learn。

【learn 模式要求】
1. 生成 3 个学习模块，从基础到进阶
2. 每个模块包含 title、content、duration_minutes
3. 模块内容具体且有实操性

【qa 模式要求】
1. topic 设为用户问题中涉及的主题，用于后续检索
2. modules 留空（不需要学习计划）

【输出格式】只输出 JSON，不要其他任何内容：
{{"mode": "learn或qa", "topic": "主题名", "goals": ["目标1", "目标2"], "modules": [{{"title": "模块标题", "content": "模块内容", "duration_minutes": 整数}}], "estimated_total_minutes": 整数}}"""

# ============================================================
# REFLECT 节点模板
# ============================================================

REFLECT_EVAL = """你是一个 Agent 系统的反思评估模块。

根据执行结果和错误信息，评估本次任务质量并决定下一步行动。

【用户输入】
{user_input}

【工具执行结果】
{tool_results}

【错误信息】
{error}

【当前重试次数 / 上限】
{retry_info}

【规则引擎预分析】
{rule_context}

【历史经验参考（仅供参考，不替代规则判断）】
{experience_context}

【要求】
1. 评估整体满意度（is_satisfactory：结果是否正确完整）
2. 如果 error 不为空或工具执行失败，分析根因（error_root_cause）
3. 决定 next_action：
   - "end"：结果满意，或已达重试上限，或问题无法通过重试解决
   - "retry"：结果不满意但仍可修复，需指定 retry_target_node
4. retry_target_node 可选值：PLANNER（重新规划）/ RETRIEVE（重新检索）/ EXECUTE（重新执行）
5. confidence 表示你对评估的把握（0.0~1.0）
6. 规则引擎预分析和历史经验仅供参考，你可以覆盖，但必须在 improvement_suggestion 中说明理由

【输出格式】只输出 JSON，不要其他任何内容：
{{"confidence": 0.0, "is_satisfactory": true, "error_root_cause": null, "improvement_suggestion": "改进建议或null", "next_action": "end或retry", "retry_target_node": "PLANNER/RETRIEVE/EXECUTE或null", "hallucination_flag": false}}"""

# ============================================================
# EXECUTE 节点模板（QA 强约束答案合成）
# ============================================================

EXECUTE_QA_SYNTHESIS = """你是一个基于知识的问答助手。

根据检索到的文档内容回答用户的问题。**你只能使用以下文档中的信息，不得使用任何外部知识。**

【用户问题】
{user_input}

【检索到的文档】
{retrieved_context}

【要求】
1. 回答必须基于上述文档内容，每条关键信息标注来源文档编号
2. 如果文档信息不足以回答问题，明确说明"基于现有资料无法完整回答"
3. 如果文档中有矛盾信息，指出矛盾并给出两种观点
4. 回答要直接、简洁，不要添加文档之外的信息

【输出格式】只输出 JSON，不要其他任何内容：
{{"answer": "回答内容（可用 [doc_N] 标注来源）", "answer_source": "rag", "sources": [1, 2, ...]}}"""

# ============================================================
# QUERY REWRITE 模板
# ============================================================

QUERY_REWRITE = """你是一个查询改写模块。

根据对话历史，将用户的省略了主语或使用了代词的查询改写为完整、独立、可直接用于向量检索的查询语句。只输出改写后的查询，不要加引号、不要解释。

【对话历史】
{conversation_history}

【当前查询】
{query}

改写后的查询："""

# ============================================================
# REFLECT 节点模板（QA 质量评估）
# ============================================================

REFLECT_QA_EVAL = """你是一个 QA 质量评估模块。

评估 AI 回答的质量，只关注两个维度：groundedness（是否有据可查）和 completeness（是否覆盖完整）。**不要判断答案的"正确性"**——你只需要判断答案的每句话是否都能在文档中找到支撑，以及用户问题的每个子问题是否都被覆盖。

【用户问题】
{user_input}

【检索到的文档】
{retrieved_context}

【AI 回答】
{final_output}

【答案来源】
{answer_source}

【当前重试次数 / 上限】
{retry_info}

【规则引擎预分析】
{rule_context}

【历史经验参考（仅供参考，不替代规则判断）】
{experience_context}

【要求】
1. groundedness_score（0.0~1.0）：答案中多少比例的信息可以在检索文档中找到直接支撑？如果答案说"文档中没有相关信息"且属实，groundedness 可以提高
2. completeness_score（0.0~1.0）：用户问题的每个子问题都有回答吗？有没有遗漏？
3. 如果 groundedness < 0.5 或 completeness < 0.5，考虑 retry
4. next_action：
   - "end"：质量合格，或已达重试上限
   - "retry"：需要重试，指定 retry_target_node（RETRIEVE 重新检索 / EXECUTE 重新生成）
5. retry_reason：如果 next_action 为 retry，说明原因（如"文档相关性不足"、"回答未覆盖用户的核心问题"）
6. 规则引擎预分析和历史经验仅供参考，你可以覆盖，但必须在 retry_reason 中说明理由

【输出格式】只输出 JSON，不要其他任何内容：
{{"groundedness_score": 0.0, "completeness_score": 0.0, "confidence": 0.0, "is_satisfactory": true, "next_action": "end或retry", "retry_target_node": "RETRIEVE/EXECUTE或null", "retry_reason": "重试原因或null"}}"""

# ============================================================
# 变量替换
# ============================================================

def render(template_name: str, **kwargs) -> str:
    """
    用变量填充 Prompt 模板
    将提示词模版里面的 {变量名} 根据**kwargs替换成对应的值
    返回完整的 Prompt 字符串
    Args:
        template_name: 模板名，如 "decide_tool_selection" / "decide_arguments"
        **kwargs: 模板中 {变量名} 对应的值

    Returns:
        填充后的完整 Prompt 字符串
    """
    templates = {
        "decide_tool_selection": DECIDE_TOOL_SELECTION,
        "decide_arguments": DECIDE_ARGUMENTS,
        "planner_plan": PLANNER_PLAN,
        "reflect_eval": REFLECT_EVAL,
        "execute_qa_synthesis": EXECUTE_QA_SYNTHESIS,  
        "reflect_qa_eval": REFLECT_QA_EVAL,
        "query_rewrite": QUERY_REWRITE,
    }

    template = templates.get(template_name)
    if template is None:
        raise ValueError(f"未知模板: {template_name}，可用: {list(templates.keys())}")

    return template.format(**kwargs)
# 功能：实现提示词模版复用，
# render模式实现了根据模板名称和传入的变量动态生成完整的提示词字符串，
# 方便在不同节点调用时使用统一的模版结构，同时又能灵活替换其中的内容。

# ============================================================
# 辅助函数
# ============================================================

def format_tools_list(tools: list[dict]) -> str:
    """
    将 registry.list_tools() 返回的列表格式化为 Prompt 可读文本

    Args:
        tools: registry.list_tools() 的输出

    Returns:
        可嵌入 Prompt 的工具列表字符串
    """
    lines = []
    for t in tools:
        params = json.dumps(t["parameters"], ensure_ascii=False, indent=2)
        #json格式化参数，保证在 Prompt 中可读性
        lines.append(f"- name: {t['name']}\n  description: {t['description']}\n  parameters: {params}")
        #拼接格式化的工具列表
    return "\n".join(lines)
#功能是将工具列表格式化为适合在提示词中展示的文本形式，
#能让LLM清晰明白哪些工具可用，怎么用每个工具

def format_parameters_schema(params: dict) -> str:
    """
    将单个工具的参数字典格式化为 Prompt 可读文本

    Args:
        params: tool["parameters"] 字典

    Returns:
        可嵌入 Prompt 的参数说明字符串
    """
    return json.dumps(params, ensure_ascii=False, indent=2)
#功能是将单个工具的参数字典格式化为适合在提示词中展示的文本形式，
#能让LLM清晰明白每个工具的参数怎么用