"""
ARneuro 配置管理器
支持YAML配置文件和环境变量
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigManager:
    """配置管理器类"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
    def _get_default_config_path(self) -> str:
        """获取默认配置文件路径"""
        # 首先检查当前目录
        current_dir = Path.cwd()
        config_files = [
            current_dir / "config.yaml",
            current_dir / "config.yml",
            current_dir / "arneuro_config.yaml",
        ]
        
        for config_file in config_files:
            if config_file.exists():
                return str(config_file)
        
        # 如果没有找到，使用默认配置
        return str(current_dir / "config.yaml")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_path = Path(self.config_path)
        
        # 如果配置文件不存在，创建默认配置
        if not config_path.exists():
            return self._create_default_config()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 应用环境变量覆盖
            config = self._apply_env_overrides(config)
            
            return config or {}
        except Exception as e:
            print(f"警告: 加载配置文件失败: {e}")
            return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        default_config = {
            "pdf_download": {
                "output_dir": "./data/pdfs",
                "max_retries": 3,
                "timeout": 30,
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "finders": [
                    "generic_citation",
                    "pubmed_central",
                    "acs",
                    "nejm",
                    "science_direct"
                ]
            },
            "ocr_processing": {
                "backend": "local",  # local 或 api
                "model_path": "/storage/work/wuguowei/Bigmodel/GLM-OCR",
                "device": "cuda",  # 或 "cpu"
                "batch_size": 4,
                "output_dir": "./data/markdown",
                "glmocr_cli_path": "glmocr",  # glmocr命令行工具路径
                "language": "ch+en",  # 支持的语言
                "api_key": "",
                "api_base_url": "https://open.bigmodel.cn/api/paas/v4/layout_parsing",
                "api_model": "glm-ocr",
                "api_timeout": 300,
                "api_use_base64": True,
                "api_return_crop_images": False,
                "api_need_layout_visualization": False,
                "api_start_page_id": None,
                "api_end_page_id": None,
                "api_user_id": ""
            },
            "document_segmentation": {
                "required_sections": ["methods", "results"],
                "optional_sections": [
                    "title", "abstract", "introduction",
                    "discussion", "conclusion", "references",
                    "acknowledgements"
                ],
                "validation_strict": False
            },
            "paths": {
                "data_dir": "./data",
                "logs_dir": "./logs",
                "cache_dir": "./cache",
                "temp_dir": "./temp"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "./logs/arneuro.log"
            }
        }
        
        # 保存默认配置到文件
        config_path = Path(self.config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"已创建默认配置文件: {config_path}")
        return default_config
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """应用环境变量覆盖"""
        env_mappings = {
            "ARNEURO_OCR_BACKEND": ["ocr_processing", "backend"],
            "ARNEURO_OCR_MODEL_PATH": ["ocr_processing", "model_path"],
            "ARNEURO_GLM_API_KEY": ["ocr_processing", "api_key"],
            "ARNEURO_PDF_OUTPUT_DIR": ["pdf_download", "output_dir"],
            "ARNEURO_DATA_DIR": ["paths", "data_dir"],
            "ARNEURO_LOG_LEVEL": ["logging", "level"]
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                # 遍历配置路径并设置值
                current = config
                for key in config_path[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[config_path[-1]] = env_value
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点分隔符，如 "pdf_download.output_dir"
            default: 默认值
        
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点分隔符
            value: 配置值
        """
        keys = key.split('.')
        current = self.config
        
        for i, k in enumerate(keys[:-1]):
            if k not in current or not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            path: 文件路径，如果为None则使用当前路径
        """
        save_path = Path(path or self.config_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()

    def load_config(self) -> Dict[str, Any]:
        """
        向后兼容接口：返回当前配置字典。

        说明：
        旧示例中通过 `ConfigManager().load_config()` 获取配置，
        新版本直接读取 `config_manager.config`，此方法保留以避免脚本失效。
        """
        return self.config
    
    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """支持字典式设置"""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """检查配置键是否存在"""
        return self.get(key) is not None


# 全局配置实例
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        ConfigManager实例
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    elif config_path is not None and str(_config_instance.config_path) != str(config_path):
        # 显式传入了新配置路径时，刷新全局实例，避免读取旧缓存配置
        _config_instance = ConfigManager(config_path)
    
    return _config_instance


def reset_config() -> None:
    """重置全局配置实例（测试或多配置切换时使用）。"""
    global _config_instance
    _config_instance = None
