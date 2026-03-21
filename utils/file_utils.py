"""
文件工具函数
"""

import os
import hashlib
import re
from pathlib import Path
from typing import Optional, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        path: 目录路径
    
    Returns:
        Path对象
    """
    path_obj = Path(path) if isinstance(path, str) else path
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """
    清理文件名，移除非法字符
    
    Args:
        filename: 原始文件名
        max_length: 最大长度
    
    Returns:
        清理后的文件名
    """
    # 移除非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # 移除控制字符
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    # 限制长度
    if len(filename) > max_length:
        # 保留扩展名
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext)]
        filename = name + ext
    
    return filename.strip()


def get_file_hash(file_path: Union[str, Path], algorithm: str = "md5") -> str:
    """
    计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法 (md5, sha1, sha256)
    
    Returns:
        文件哈希值
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    hash_func = getattr(hashlib, algorithm)()
    
    with open(file_path, 'rb') as f:
        # 分块读取大文件
        for chunk in iter(lambda: f.read(4096), b''):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def get_file_size(file_path: Union[str, Path]) -> int:
    """
    获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小（字节）
    """
    file_path = Path(file_path)
    return file_path.stat().st_size if file_path.exists() else 0


def is_valid_pdf(file_path: Union[str, Path]) -> bool:
    """
    检查文件是否为有效的PDF
    
    Args:
        file_path: 文件路径
    
    Returns:
        是否为有效PDF
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    # 检查文件大小
    if file_path.stat().st_size < 100:  # PDF文件至少100字节
        return False
    
    # 检查文件头
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            return header == b'%PDF-'
    except:
        return False


def copy_file_with_progress(src: Union[str, Path], dst: Union[str, Path], 
                           chunk_size: int = 8192) -> None:
    """
    复制文件并显示进度
    
    Args:
        src: 源文件路径
        dst: 目标文件路径
        chunk_size: 块大小
    """
    src_path = Path(src)
    dst_path = Path(dst)
    
    if not src_path.exists():
        raise FileNotFoundError(f"源文件不存在: {src_path}")
    
    total_size = src_path.stat().st_size
    copied = 0
    
    ensure_dir(dst_path.parent)
    
    with open(src_path, 'rb') as src_file, open(dst_path, 'wb') as dst_file:
        while True:
            chunk = src_file.read(chunk_size)
            if not chunk:
                break
            
            dst_file.write(chunk)
            copied += len(chunk)
            
            # 计算进度百分比
            progress = (copied / total_size) * 100
            print(f"\r复制进度: {progress:.1f}%", end='')
    
    print()  # 换行


def find_files_by_pattern(directory: Union[str, Path], pattern: str) -> list:
    """
    按模式查找文件
    
    Args:
        directory: 目录路径
        pattern: 文件模式（支持通配符）
    
    Returns:
        文件路径列表
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    return list(directory.rglob(pattern))


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    获取文件扩展名（小写）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件扩展名
    """
    file_path = Path(file_path)
    return file_path.suffix.lower()


def create_temp_file(content: bytes, suffix: str = ".tmp") -> Path:
    """
    创建临时文件
    
    Args:
        content: 文件内容
        suffix: 文件后缀
    
    Returns:
        临时文件路径
    """
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as f:
        f.write(content)
        return Path(f.name)


def cleanup_temp_files(temp_files: list) -> None:
    """
    清理临时文件
    
    Args:
        temp_files: 临时文件路径列表
    """
    for temp_file in temp_files:
        try:
            Path(temp_file).unlink(missing_ok=True)
        except:
            pass