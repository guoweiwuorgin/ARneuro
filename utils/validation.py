"""
数据验证函数
"""

import re
from pathlib import Path
from typing import Optional, Tuple


def validate_pmid(pmid: str) -> Tuple[bool, Optional[str]]:
    """
    验证PMID格式
    
    Args:
        pmid: PMID字符串
    
    Returns:
        (是否有效, 错误信息)
    """
    if not pmid:
        return False, "PMID不能为空"
    
    # 移除可能的空格
    pmid = str(pmid).strip()
    
    # 检查是否为数字
    if not pmid.isdigit():
        return False, f"PMID必须为数字: {pmid}"
    
    # 检查长度（PMID通常是8位数字）
    if len(pmid) < 5 or len(pmid) > 10:
        return False, f"PMID长度异常: {pmid}"
    
    return True, None


def validate_pdf(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    验证PDF文件
    
    Args:
        file_path: PDF文件路径
    
    Returns:
        (是否有效, 错误信息)
    """
    if not file_path.exists():
        return False, f"文件不存在: {file_path}"
    
    # 检查文件大小
    file_size = file_path.stat().st_size
    if file_size < 100:  # PDF文件至少100字节
        return False, f"文件过小: {file_size}字节"
    
    if file_size > 100 * 1024 * 1024:  # 100MB
        return False, f"文件过大: {file_size / (1024*1024):.1f}MB"
    
    # 检查文件头
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                return False, "不是有效的PDF文件"
    except Exception as e:
        return False, f"读取文件失败: {str(e)}"
    
    return True, None


def validate_markdown(file_path: Path) -> Tuple[bool, Optional[str]]:
    """
    验证Markdown文件
    
    Args:
        file_path: Markdown文件路径
    
    Returns:
        (是否有效, 错误信息)
    """
    if not file_path.exists():
        return False, f"文件不存在: {file_path}"
    
    # 检查文件大小
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "文件为空"
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return False, f"文件过大: {file_size / (1024*1024):.1f}MB"
    
    # 检查文件内容
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(1000)  # 读取前1000个字符
            
            if not content.strip():
                return False, "文件内容为空"
            
            # 检查是否包含一些文本内容
            text_chars = sum(1 for c in content if c.isalpha() or c.isspace())
            if text_chars < 10:
                return False, "文件内容过少"
    except UnicodeDecodeError:
        return False, "文件编码不是UTF-8"
    except Exception as e:
        return False, f"读取文件失败: {str(e)}"
    
    return True, None


def validate_csv_file(file_path: Path, required_columns: list = None) -> Tuple[bool, Optional[str]]:
    """
    验证CSV文件
    
    Args:
        file_path: CSV文件路径
        required_columns: 必需的列名
    
    Returns:
        (是否有效, 错误信息)
    """
    if not file_path.exists():
        return False, f"文件不存在: {file_path}"
    
    # 检查文件扩展名
    if file_path.suffix.lower() not in ['.csv', '.tsv', '.txt']:
        return False, f"不支持的文件格式: {file_path.suffix}"
    
    # 检查文件大小
    file_size = file_path.stat().st_size
    if file_size == 0:
        return False, "文件为空"
    
    # 尝试读取文件
    try:
        import pandas as pd
        
        # 尝试读取CSV
        try:
            df = pd.read_csv(file_path)
        except:
            # 尝试TSV
            df = pd.read_csv(file_path, sep='\t')
        
        # 检查是否有数据
        if df.empty:
            return False, "CSV文件没有数据"
        
        # 检查必需列
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return False, f"缺少必需列: {missing_columns}"
        
        # 检查PMID列（如果存在）
        if 'PMID' in df.columns:
            # 检查PMID有效性
            pmids = df['PMID'].dropna().astype(str)
            invalid_pmids = []
            
            for pmid in pmids:
                valid, _ = validate_pmid(pmid)
                if not valid:
                    invalid_pmids.append(pmid)
            
            if invalid_pmids:
                return False, f"无效的PMID: {invalid_pmids[:5]}"
    
    except ImportError:
        # 如果没有pandas，进行简单检查
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                if len(lines) < 2:
                    return False, "CSV文件至少需要一行标题和一行数据"
                
                # 检查是否有PMID列
                header = lines[0].strip().lower()
                if 'pmid' not in header and required_columns:
                    return False, "CSV文件缺少PMID列"
        except Exception as e:
            return False, f"读取CSV文件失败: {str(e)}"
    
    except Exception as e:
        return False, f"验证CSV文件失败: {str(e)}"
    
    return True, None


def validate_output_dir(dir_path: Path, create_if_missing: bool = True) -> Tuple[bool, Optional[str]]:
    """
    验证输出目录
    
    Args:
        dir_path: 目录路径
        create_if_missing: 如果目录不存在是否创建
    
    Returns:
        (是否有效, 错误信息)
    """
    if dir_path.exists():
        if not dir_path.is_dir():
            return False, f"路径不是目录: {dir_path}"
        
        # 检查目录是否可写
        try:
            test_file = dir_path / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            return False, f"目录不可写: {str(e)}"
    
    elif create_if_missing:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            return False, f"创建目录失败: {str(e)}"
    
    else:
        return False, f"目录不存在: {dir_path}"
    
    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    验证URL格式
    
    Args:
        url: URL字符串
    
    Returns:
        (是否有效, 错误信息)
    """
    if not url:
        return False, "URL不能为空"
    
    url_pattern = re.compile(
        r'^(https?|ftp)://'  # 协议
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # 域名
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # IP地址
        r'(?::\d+)?'  # 端口
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not re.match(url_pattern, url):
        return False, f"无效的URL格式: {url}"
    
    return True, None


def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    验证邮箱格式
    
    Args:
        email: 邮箱地址
    
    Returns:
        (是否有效, 错误信息)
    """
    if not email:
        return False, "邮箱不能为空"
    
    email_pattern = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    
    if not re.match(email_pattern, email):
        return False, f"无效的邮箱格式: {email}"
    
    return True, None


def validate_file_permissions(file_path: Path, required_permissions: str = "rw") -> Tuple[bool, Optional[str]]:
    """
    验证文件权限
    
    Args:
        file_path: 文件路径
        required_permissions: 所需权限 (r:读, w:写, x:执行)
    
    Returns:
        (是否有效, 错误信息)
    """
    if not file_path.exists():
        return False, f"文件不存在: {file_path}"
    
    errors = []
    
    if 'r' in required_permissions:
        if not os.access(file_path, os.R_OK):
            errors.append("不可读")
    
    if 'w' in required_permissions:
        if not os.access(file_path, os.W_OK):
            errors.append("不可写")
    
    if 'x' in required_permissions:
        if not os.access(file_path, os.X_OK):
            errors.append("不可执行")
    
    if errors:
        return False, f"文件权限不足: {', '.join(errors)}"
    
    return True, None