import os
import sys
from concurrent.futures import ThreadPoolExecutor

# 确保能找到根目录模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mod_base import Module
from ai_module import mod_general
from log import log
from pydantic import BaseModel, Field

class BackendConfig(BaseModel):
    # 所有 backend 通用的基础配置
    pass

class BackendBase(Module):
    """
    Backend 组件基类。
    支持从 _shared_backend 穿透配置，并自动初始化对应的 mod_general。
    """
    shared_namespace = "_shared_backend"
    config_model = BackendConfig
    
    def __init__(self, external_config: dict, context: dict, pool: ThreadPoolExecutor, namespace: str):
        self.namespace = namespace
        super().__init__(context)
        self.pool = pool
        
        # 1. 加载 backend 自身的逻辑配置 (如 build_dir, source_dir 等)
        self.backend_cfg = self.cfg(external_config)
        
        # 2. 自动初始化一个同命名空间的 AI 模块
        # 这使得 backend 逻辑可以直接使用 self.mod 进行 AI 调用
        self.mod = mod_general.Module(external_config, context, pool, namespace=namespace)
        
        # 记录初始化信息
        ai_model = getattr(self.mod.sub_module.config, "model", "unknown")
        log(f"[{self.namespace}] Backend initialized (AI: {ai_model})")

    def get_cfg(self, key, default=None):
        """便捷获取 backend 配置项"""
        return self.backend_cfg.get(key, default)
