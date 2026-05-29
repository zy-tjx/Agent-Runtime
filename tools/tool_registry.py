"""
工具注册器
负责工具的注册、发现和查询，采用显式注册模式（非目录扫描）
这段代码实现了一个工具注册与管理系统，
专为 AI Agent 架构设计，核心目标是结构化管理可调用工具
采用显示注册方式，避免了目录扫描的复杂性和性能问题
"""
from typing import Any, Callable, Optional
from pydantic import BaseModel


class ToolInfo:
    """单个工具的元信息"""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: type[BaseModel],
        run_func: Callable[[Any], Any],
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.run_func = run_func


class ToolRegistry:
    """
    工具注册表

    用法:
        registry = ToolRegistry()
        registry.register("search_docs", "检索知识库", SearchDocsInput, run)
        tool = registry.get("search_docs")
        tools = registry.list_tools()
    """

    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(
        self,
        name: str,
        description: str,
        input_schema: type[BaseModel],  #工具输入参数的 Pydantic 模型类
        run_func: Callable[[Any], Any], #工具的执行函数，接受 input_schema 定义的参数，返回 ToolOutput
    ) -> None:
        """注册一个工具"""
        if name in self._tools:
            raise ValueError(f"工具 '{name}' 已存在，不允许重复注册")
        self._tools[name] = ToolInfo(name, description, input_schema, run_func)

    def get(self, name: str) -> Optional[ToolInfo]:
        """根据名称获取工具，不存在返回 None"""
        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        """检查工具是否已注册"""
        return name in self._tools

    def list_tools(self) -> list[dict[str, Any]]:
        """
        列出所有已注册工具的基本信息
        供 DECIDE 节点获取可用工具列表
        """
        result = []
        for tool in self._tools.values():
            # 从 Pydantic 模型中提取参数字段信息
            params = {}
            for field_name, field_info in tool.input_schema.model_fields.items():
                params[field_name] = {
                    "type": str(field_info.annotation),
                    "required": field_info.is_required(),
                    "default": field_info.default if not field_info.is_required() else None,
                }
            result.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": params,
                }
            )
        return result

    def __len__(self) -> int:
        return len(self._tools)


def create_default_registry() -> ToolRegistry:
    """
    创建预注册 Runtime 工具的注册表

    供 DECIDE、EXECUTE 等节点使用，避免各节点重复维护注册逻辑。
    """
    from tools.search_docs import SearchDocsInput, run as search_run
    from tools.memory_write import MemoryWriteInput, run as mw_run

    registry = ToolRegistry()
    registry.register(
        "search_docs", "从知识库中检索相关文档", SearchDocsInput, search_run
    )
    registry.register(
        "memory_write", "写入长期记忆（用户画像/进度/经验）", MemoryWriteInput, mw_run
    )
    return registry
