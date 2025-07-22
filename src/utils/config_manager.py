import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """配置管理器，負責載入和管理所有配置檔案"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """載入所有配置檔案"""
        if not self.config_dir.exists():
            raise FileNotFoundError(f"配置目錄不存在: {self.config_dir}")
        
        yaml_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        
        for yaml_file in yaml_files:
            config_name = yaml_file.stem
            try:
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    self.configs[config_name] = yaml.safe_load(f)
                logging.info(f"載入配置檔案: {yaml_file}")
            except Exception as e:
                logging.error(f"載入配置檔案失敗 {yaml_file}: {e}")
    
    def get_config(self, config_name: str) -> Dict[str, Any]:
        """獲取指定的配置"""
        if config_name not in self.configs:
            raise KeyError(f"找不到配置: {config_name}")
        return self.configs[config_name]
    
    def get_nested_config(self, config_name: str, *keys) -> Any:
        """獲取巢狀配置值"""
        config = self.get_config(config_name)
        for key in keys:
            if key not in config:
                raise KeyError(f"找不到配置項: {config_name}.{'.'.join(keys)}")
            config = config[key]
        return config
    
    def update_config(self, config_name: str, key: str, value: Any):
        """更新配置值"""
        if config_name not in self.configs:
            self.configs[config_name] = {}
        self.configs[config_name][key] = value
    
    def save_config(self, config_name: str, overwrite: bool = False):
        """儲存配置到檔案"""
        config_path = self.config_dir / f"{config_name}.yaml"
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"配置檔案已存在: {config_path}")
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.configs[config_name], f, default_flow_style=False, allow_unicode=True)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """獲取所有配置"""
        return self.configs
    
    def merge_configs(self, *config_names) -> Dict[str, Any]:
        """合併多個配置"""
        merged = {}
        for config_name in config_names:
            merged.update(self.get_config(config_name))
        return merged

# 使用範例
if __name__ == "__main__":
    # 初始化配置管理器
    config_manager = ConfigManager()
    
    # 獲取模型配置
    model_config = config_manager.get_config("model_config")
    print("模型配置:", model_config)
    
    # 獲取特定配置項
    batch_size = config_manager.get_nested_config("training_config", "training", "batch_size")
    print("Batch size:", batch_size)
    
    # 獲取訓練配置
    training_config = config_manager.get_config("training_config")
    print("訓練配置:", training_config)