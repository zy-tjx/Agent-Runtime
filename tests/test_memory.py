"""
Memory 系统测试
覆盖：long_term_memory / memory_write / memory_read / REFLECT 单写源
"""
import sqlite3
import pytest

from memory.long_term_memory import MemoryRecord, save, load, DB_PATH
from tools.tool_registry import create_default_registry
from tools.tool_executor import ToolExecutor


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def clean_test_data():
    """每个测试前后清理数据库中的测试数据"""
    yield
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM memories WHERE key LIKE 'test_%' OR key LIKE 'reflect_%'")
        conn.commit()
        conn.close()
    except Exception:
        pass


# ============================================================
# 1. MemoryRecord Schema 测试
# ============================================================

class TestMemoryRecord:
    def test_default_values(self):
        record = MemoryRecord(key="test_1", value={"score": 0.9})
        assert record.category == "experience"
        assert record.timestamp == ""
        assert record.session_id is None

    def test_explicit_category(self):
        record = MemoryRecord(key="test_2", value={}, category="profile")
        assert record.category == "profile"

    def test_invalid_category_accepted_by_model(self):
        """Pydantic 不做枚举校验（枚举校验在 MemoryWriteInput 层）"""
        record = MemoryRecord(key="test_3", value={}, category="invalid")
        assert record.category == "invalid"


# ============================================================
# 2. long_term_memory 存储层测试
# ============================================================

class TestLongTermMemory:
    def test_save_and_load_by_key(self):
        save(MemoryRecord(key="test_save_1", value={"data": "hello"}, category="profile"))
        rows = load(key="test_save_1")
        assert len(rows) == 1
        assert rows[0]["value"]["data"] == "hello"

    def test_load_by_category(self):
        save(MemoryRecord(key="test_cat_1", value={"n": 1}, category="profile"))
        save(MemoryRecord(key="test_cat_2", value={"n": 2}, category="experience"))
        rows = load(category="profile")
        assert len(rows) >= 1
        assert all(r["category"] == "profile" for r in rows)

    def test_load_missing_key(self):
        rows = load(key="test_nonexistent_xyz")
        assert rows == []

    def test_save_overwrite(self):
        save(MemoryRecord(key="test_overwrite", value={"v": 1}))
        save(MemoryRecord(key="test_overwrite", value={"v": 2}))
        rows = load(key="test_overwrite")
        assert rows[0]["value"]["v"] == 2

    def test_load_all(self):
        save(MemoryRecord(key="test_all_1", value={}, category="session"))
        save(MemoryRecord(key="test_all_2", value={}, category="profile"))
        rows = load()
        assert len(rows) >= 2


# ============================================================
# 3. memory_write / memory_read 工具测试（通过 Executor）
# ============================================================

class TestMemoryTools:
    @pytest.fixture
    def executor(self):
        return ToolExecutor(create_default_registry())

    def test_write_success(self, executor):
        out = executor.execute("memory_write", {
            "key": "test_tool_write",
            "value": {"topic": "RAG", "confidence": 0.9},
            "category": "experience",
        })
        assert out.status == "success"
        rows = load(key="test_tool_write")
        assert rows[0]["value"]["topic"] == "RAG"

    def test_write_invalid_category(self, executor):
        out = executor.execute("memory_write", {
            "key": "test_bad",
            "value": {},
            "category": "bad_value",
        })
        assert out.status == "failed"


# ============================================================
# 4. REFLECT 单写源测试
# ============================================================

class TestReflectWriteThrough:
    def test_reflect_qa_writes_to_memory(self):
        """REFLECT QA 执行后数据库应有记录"""
        import os
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node

        state = {
            "mode": "qa",
            "user_input": "什么是 RAG",
            "retrieved_context": [
                {"doc_id": "1", "content": "RAG 是检索增强生成的简称",
                 "source": "test.md", "score": 0.8}
            ],
            "final_output": "RAG 是检索增强生成技术",
            "answer_source": "rag",
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        assert result["reflection"]["next_action"] is not None

        # 验证数据库写入
        rows = load(category="experience")
        reflect_rows = [r for r in rows if "reflect_" in r["key"]]
        assert len(reflect_rows) >= 1
        last = reflect_rows[-1]
        assert "reflection" in last["value"]

    def test_reflect_learn_writes_to_memory(self):
        """REFLECT learn 执行后数据库也应有记录"""
        import os
        os.environ.pop("DECIDE_USE_KEYWORD", None)
        from runtime.node_reflect import reflect_node

        state = {
            "mode": "learn",
            "user_input": "学习 Agent",
            "tool_results": [
                {"tool_name": "search_docs", "status": "success",
                 "result": {"documents": [{"doc_id": "1"}]}}
            ],
            "error": None,
            "retry_count": 0,
            "max_retries": 3,
        }
        result = reflect_node(state)
        assert result["reflection"]["next_action"] is not None

        rows = load(category="experience")
        reflect_rows = [r for r in rows if "reflect_" in r["key"]]
        assert len(reflect_rows) >= 1
