"""
文档加载器
从指定目录加载 MD 文件，解析 YAML 风格元数据头部
"""
import os

def _read_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
    
def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    解析 --- 包围的 YAML 风格元数据头部

    Args:
        text: 完整文件内容

    Returns:
        (metadata_dict, body_text)
    """
    lines = text.split("\n")
    metadata = {}
    body_start = 0

    if lines and lines[0].strip() == "---":
        # 找第二个 --- 的位置
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                body_start = i + 1
                break
        # 解析中间行（兼容中英文冒号）
        for line in lines[1:body_start - 1]:
            sep = "：" if "：" in line else ":"
            if sep in line:
                key, _, value = line.partition(sep)
                key = key.strip()
                value = value.strip()
                if key in ("关键词", "前置知识"):
                    # 逗号分隔的列表字段
                    metadata[key] = [v.strip() for v in value.split(",") if v.strip()]
                else:
                    metadata[key] = value

    body = "\n".join(lines[body_start:])
    return metadata, body

def load_documents(directory: str) -> list[dict]:
    """
    加载目录下所有 .md 文件

    Args:
        directory: 知识库目录路径

    Returns:
        [{filename, content, metadata: {topic, difficulty, keywords, prerequisites}}]
    """
    documents = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(directory, filename)
        content = _read_file(filepath)
        metadata, body = _parse_frontmatter(content)
        #body是正文
        #metadata是头部元数据

        documents.append(
            {
                "filename": filename,
                "content": body.strip(),
                "metadata": metadata,
            }
        )
    return documents






