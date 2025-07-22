import os
from pathlib import Path
from .config_manager import ConfigManager

# 全域配置管理器
_config_manager = None

def get_config_manager() -> ConfigManager:
    """獲取全域配置管理器"""
    global _config_manager
    if _config_manager is None:
        config_dir = os.getenv('CONFIG_DIR', 'configs')
        _config_manager = ConfigManager(config_dir)
    return _config_manager

def load_model_config():
    """載入模型配置"""
    return get_config_manager().get_config("model_config")

def load_training_config():
    """載入訓練配置"""
    return get_config_manager().get_config("training_config")

def load_processing_config():
    """載入處理配置"""
    return get_config_manager().get_config("processing_config")

def get_model_name():
    """獲取模型名稱"""
    return get_config_manager().get_nested_config("model_config", "model", "name")

def get_batch_size():
    """獲取批次大小"""
    return get_config_manager().get_nested_config("training_config", "training", "batch_size")

def get_output_dir():
    """獲取輸出目錄"""
    return get_config_manager().get_nested_config("training_config", "paths", "output_dir")