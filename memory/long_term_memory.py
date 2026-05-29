"""
长期记忆存储层
纯存取：追加（save） + 查询（load）

数据库文件：项目根目录 memory/long_term.db
表结构：key TEXT PK | value TEXT(JSON) | category TEXT | timestamp TEXT | session_id TEXT

写入口限制：仅 REFLECT 节点写入（write-through，单写源原则）
读入口：PLANNER / DECIDE / REFLECT 均可读取
"""
import json
import sqlite3
import os
import time
from typing import Optional
from pydantic import BaseModel, Field

# ── 数据库路径 ──
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "memory")
DB_PATH = os.path.join(DB_DIR, "long_term.db")


def _ensure_db_dir():
    os.makedirs(DB_DIR, exist_ok=True)
#确保数据库目录存在

def _init_table(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT 'experience',
            timestamp TEXT NOT NULL,
            session_id TEXT
        )
    """)
    conn.commit()

def _get_connection() -> sqlite3.Connection:
    _ensure_db_dir()    #创建数据库目录（如果不存在）
    conn = sqlite3.connect(DB_PATH)    
    conn.row_factory = sqlite3.Row  #创建数据库文件
    _init_table(conn)   #建表
    return conn
# 获取数据库连接，并确保表结构存在

# ============================================================
# 统一 Schema
# ============================================================

class MemoryRecord(BaseModel):
    """所有记忆条目的统一结构"""
    key: str = Field(description="唯一键（如 reflect_20260516_001）")
    value: dict = Field(description="记忆内容，JSON 可序列化")
    category: str = Field(
        default="experience",
        description="记忆类别：profile（用户画像）/ experience（反思经验）"
    )
    timestamp: str = Field(default="", description="ISO 时间戳")
    session_id: Optional[str] = Field(default=None, description="关联会话 ID")


# ============================================================
# 核心操作
# ============================================================

def save(record: MemoryRecord) -> bool:
    """
    写入一条记忆（追加）

    Args:
        record: MemoryRecord 实例

    Returns:
        True 表示写入成功

    Raises:
        ValueError: 参数校验失败
    """
    if record.timestamp == "":
        record.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

    conn = _get_connection()
    #连接数据库
    conn.execute(
        """INSERT OR REPLACE INTO memories (key, value, category, timestamp, session_id)
           VALUES (?, ?, ?, ?, ?)""",
        (
            record.key,
            json.dumps(record.value, ensure_ascii=False),
            record.category,
            record.timestamp,
            record.session_id,
        ),
    )
    conn.commit()
    conn.close()
    return True

def _row_to_dict(row) -> dict:
    return {
        "key": row["key"],
        "value": json.loads(row["value"]),
        "category": row["category"],
        "timestamp": row["timestamp"],
        "session_id": row["session_id"],
    }

def load(
    key: Optional[str] = None,
    category: Optional[str] = None,
) -> list[dict]:
    """
    查询记忆

    Args:
        key: 精确匹配 key（可选）
        category: 按类别筛选（可选）

    Returns:
        匹配的 MemoryRecord 字典列表，无结果时返回空列表
    """
    conn = _get_connection()
    if key:
        row = conn.execute(
            "SELECT * FROM memories WHERE key = ?", (key,)
        ).fetchone()
        conn.close()
        if row is None:
            return []
        return [_row_to_dict(row)]

    if category:
        rows = conn.execute(
            "SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC",
            (category,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memories ORDER BY timestamp DESC"
        ).fetchall()

    conn.close()
    return [_row_to_dict(r) for r in rows]
# ——用户画像（提供Planner 避免重复推荐已学内容）
def get_user_profile(user_id: str = "default") -> dict:
    """获取用户已完成的主题列表"""
    records = load(key=f"profile_{user_id}")
    if records:
        return records[0]["value"]
    return {"topics_completed": []}


def mark_topic_completed(user_id: str, topic: str) -> None:
    """标记主题为已完成（fire-and-forget，失败不抛异常）"""
    profile = get_user_profile(user_id)
    if topic not in profile.get("topics_completed", []):
        profile.setdefault("topics_completed", []).append(topic)
    try:
        save(MemoryRecord(key=f"profile_{user_id}", value=profile, category="profile"))
    except Exception:
        pass


# ── 经验检索（供 REFLECT 使用） ──

def load_recent_experiences(limit: int = 5, mode: str | None = None) -> list[dict]:
    """
    获取最近的经验记录

    Args:
        limit: 返回数量上限
        mode: 可选，按模式筛选（learn / qa），为 None 时返回全部

    Returns:
        最近 N 条经验记录的列表，每条为 MemoryRecord 字典
    """
    all_records = load(category="experience")
    if mode:
        all_records = [r for r in all_records if r["value"].get("mode") == mode]
    return all_records[:limit]


def load_experience_summaries(limit: int = 5, mode: str | None = None) -> list[str]:
    """
    将最近经验记录格式化为简短文本，供 prompt 嵌入

    每条摘要格式：
      [时间] 模式=learn, 决策=end, 根因=None, 满意度=是

    Args:
        limit: 返回数量上限
        mode: 可选，按模式筛选

    Returns:
        摘要文本列表，无记录时返回 ["无历史经验"]
    """
    records = load_recent_experiences(limit=limit, mode=mode)
    if not records:
        return ["无历史经验"]

    summaries = []
    for r in records:
        v = r["value"]
        refl = v.get("reflection", {})
        ts = r["timestamp"][:16]  # 截到分钟
        if v.get("mode") == "qa":
            summaries.append(
                f"[{ts}] QA, groundedness={refl.get('groundedness_score', '?')}, "
                f"completeness={refl.get('completeness_score', '?')}, "
                f"决策={refl.get('next_action', '?')}"
            )
        else:
            summaries.append(
                f"[{ts}] learn, 决策={refl.get('next_action', '?')}, "
                f"根因={refl.get('error_root_cause') or '无'}, "
                f"满意度={refl.get('is_satisfactory', '?')}"
            )
    return summaries
