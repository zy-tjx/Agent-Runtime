"""
文本切分器
按 ## 标题为锚点语义切分，保证 chunk 语义完整
"""
import re

def _split_by_heading(text: str) -> list[tuple[str, str]]:
    """
    按 ## 标题切分文本

    Returns:
        [(heading_text, section_body), ...]
    """
    # 匹配 "## 标题" 行
    pattern = re.compile(r"^## (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        return [("", text)]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append((heading, body))

    return sections

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    将单段文本按 chunk_size 切分，相邻 chunk 有 overlap 重叠

    优先级：段落边界 > 句子边界 > 字符边界
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            # 保存当前 chunk
            if current:
                chunks.append(current)
            # 新段落：如果单段超长，强制按字符切
            if len(para) > chunk_size:
                sub_chunks = _force_split(para, chunk_size, overlap)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks


def _force_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """强制按字符切分超长段落，带 overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def split_documents(
    docs: list[dict], chunk_size: int = 500, overlap: int = 50
) -> list[dict]:
    """
    将文档按 ## 标题 + 段落边界切分为 chunks

    Args:
        docs: document_loader.load_documents() 的输出
        chunk_size: 每个 chunk 最大字符数
        overlap: 相邻 chunk 的重叠字符数

    Returns:
        [{doc_id, content, source, chunk_index, heading, metadata}]
        分别对应来源文档ID、chunk内容、来源文件名、相似度分数、所在标题、原文档元数据
    """
    chunks = []
    for doc in docs:
        filename = doc["filename"]
        content = doc["content"]
        metadata = doc.get("metadata", {})

        # ── 按 ## 标题切分 ──
        sections = _split_by_heading(content)

        for heading, section_text in sections:
            if not section_text.strip():
                continue

            # ── 按 paragraph 细切 + 按长度分块 ──
            section_chunks = _chunk_text(
                section_text, chunk_size=chunk_size, overlap=overlap
            )
            for chunk_text in section_chunks:
                chunk_id = f"{filename.replace('.md', '')}_{len(chunks)}"
                chunks.append(
                    {
                        "doc_id": chunk_id,
                        "content": chunk_text,
                        "source": filename,
                        "chunk_index": len(chunks),
                        "heading": heading,
                        "metadata": metadata,
                    }
                )
    return chunks


