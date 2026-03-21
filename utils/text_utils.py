"""
文本处理工具函数
"""

import re
from typing import List, Optional


def clean_text(text: str, remove_empty_lines: bool = True) -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
        remove_empty_lines: 是否移除空行
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    
    # 移除首尾空格
    text = text.strip()
    
    if remove_empty_lines:
        # 移除空行
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        text = '\n'.join(lines)
    
    return text


def split_into_sentences(text: str, language: str = "en") -> List[str]:
    """
    将文本分割成句子
    
    Args:
        text: 文本
        language: 语言 (en, zh, mixed)
    
    Returns:
        句子列表
    """
    if not text:
        return []
    
    text = clean_text(text)
    
    if language == "zh" or language == "mixed":
        # 中文句子分割
        sentences = re.split(r'[。！？!?；;]', text)
    else:
        # 英文句子分割
        sentences = re.split(r'[.!?;]', text)
    
    # 清理句子
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def normalize_text(text: str) -> str:
    """
    标准化文本（统一字符、移除特殊字符等）
    
    Args:
        text: 原始文本
    
    Returns:
        标准化后的文本
    """
    if not text:
        return ""
    
    # 统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 统一引号
    text = text.replace('"', '"').replace("'", "'")
    
    # 移除控制字符
    text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
    
    # 标准化空格
    text = re.sub(r'[ \t]+', ' ', text)
    
    return text.strip()


def extract_section(text: str, section_title: str, next_section: Optional[str] = None) -> str:
    """
    从文本中提取特定章节
    
    Args:
        text: 完整文本
        section_title: 章节标题
        next_section: 下一章节标题（用于确定结束位置）
    
    Returns:
        章节内容
    """
    if not text:
        return ""
    
    # 构建正则表达式
    title_pattern = re.escape(section_title)
    
    if next_section:
        next_pattern = re.escape(next_section)
        pattern = rf'{title_pattern}.*?(?={next_pattern}|\Z)'
    else:
        pattern = rf'{title_pattern}.*'
    
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        # 移除标题部分
        content = match.group(0)
        content = re.sub(rf'^{title_pattern}\s*', '', content, flags=re.IGNORECASE)
        return content.strip()
    
    return ""


def count_words(text: str) -> int:
    """
    统计文本中的单词数
    
    Args:
        text: 文本
    
    Returns:
        单词数
    """
    if not text:
        return 0
    
    # 简单单词计数（适用于英文）
    words = re.findall(r'\b\w+\b', text)
    return len(words)


def find_keywords(text: str, keywords: List[str], case_sensitive: bool = False) -> List[str]:
    """
    在文本中查找关键词
    
    Args:
        text: 文本
        keywords: 关键词列表
        case_sensitive: 是否区分大小写
    
    Returns:
        找到的关键词列表
    """
    if not text or not keywords:
        return []
    
    found = []
    
    for keyword in keywords:
        if case_sensitive:
            if keyword in text:
                found.append(keyword)
        else:
            if keyword.lower() in text.lower():
                found.append(keyword)
    
    return found


def remove_references(text: str) -> str:
    """
    移除参考文献部分
    
    Args:
        text: 文本
    
    Returns:
        移除参考文献后的文本
    """
    if not text:
        return ""
    
    # 常见的参考文献标题
    reference_patterns = [
        r'References.*',
        r'Bibliography.*',
        r'Cited Literature.*',
        r'参考文献.*',
        r'引用文献.*'
    ]
    
    for pattern in reference_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text.strip()


def extract_tables_from_markdown(markdown_text: str) -> List[str]:
    """
    从Markdown文本中提取表格
    
    Args:
        markdown_text: Markdown文本
    
    Returns:
        表格列表
    """
    if not markdown_text:
        return []
    
    # 查找Markdown表格
    table_pattern = r'(\|.*\|\n)+\|?[\s-]+\|.*\n(\|.*\|\n)+'
    tables = re.findall(table_pattern, markdown_text)
    
    # 提取完整的表格文本
    table_texts = []
    for match in re.finditer(table_pattern, markdown_text):
        table_texts.append(match.group(0))
    
    return table_texts


def detect_language(text: str) -> str:
    """
    检测文本语言
    
    Args:
        text: 文本
    
    Returns:
        语言代码 (en, zh, mixed)
    """
    if not text:
        return "en"
    
    # 采样前1000个字符
    sample = text[:1000]
    
    # 统计中文字符
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', sample)
    chinese_ratio = len(chinese_chars) / len(sample) if sample else 0
    
    # 统计英文字符
    english_chars = re.findall(r'[a-zA-Z]', sample)
    english_ratio = len(english_chars) / len(sample) if sample else 0
    
    if chinese_ratio > 0.3 and english_ratio > 0.3:
        return "mixed"
    elif chinese_ratio > english_ratio:
        return "zh"
    else:
        return "en"