import os
import importlib
from concurrent.futures import ThreadPoolExecutor
from mod_base import Module as BaseModule
from typing import Any, Optional, Dict
from log import log

class Module(BaseModule):
    """
    通用 AI 模块代理。
    
    配置示例:
    {
        "ai_general": {
            "ai": "gemini",
            "gemini": {
                "model": "gemini-1.5-pro",
                "_override": true
            }
        },
        "_shared_ai_general": { ... },
        "_shared_ai_gemini": {
            "api_key": "YOUR_SHARED_KEY"
        }
    }
    """
    namespace = "ai_general"
    shared_namespace = "_shared_ai_general"
    sub_module: Any = None

    def __init__(self, external_config: dict, context: dict, pool: ThreadPoolExecutor, namespace: str = "ai_general"):
        self.namespace = namespace
        super().__init__(context)
        
        # 1. 获取 ai_general 自身的配置 (包含与 _shared_ai_general 的合并)
        gen_cfg = self.cfg(external_config)
        
        # 2. 确定后端类型
        ai_type = gen_cfg.get("ai", "gemini").lower()
        
        # 3. 映射子模块信息
        type_map = {
            "gemini": (".mod_gemini", "ai_gemini"),
            "ollama": (".mod_ollama", "ai_ollama"),
            "openai": (".mod_openai", "ai_openai")
        }
        
        if ai_type not in type_map:
            raise ValueError(f"Unsupported AI type in ai_general: {ai_type}")
            
        mod_path, sub_ns = type_map[ai_type]
        
        # 4. 构造合成配置 (实现配置穿透)
        # 我们复制一份原始配置，确保子模块能访问到根部的 _shared_xxx
        synthetic_config = external_config.copy()
        
        # 从 ai_general 内部提取该后端的专属配置，并注入到合成配置的对应命名空间中
        # 这样子模块在调用 self.cfg(synthetic_config) 时，就能拿到这份穿透后的配置
        sub_instance_cfg = gen_cfg.get(ai_type, {})
        synthetic_config[sub_ns] = sub_instance_cfg
        
        log(f"[{self.namespace}] Proxying to {ai_type} with penetrated config from {self.namespace}.{ai_type}")
        
        # 5. 动态加载子模块
        pkg = __package__ or "ai_module"
        module = importlib.import_module(mod_path, package=pkg)
        SubModuleClass = getattr(module, "Module")
        
        # 初始化子模块
        self.sub_module = SubModuleClass(synthetic_config, context, pool, namespace=sub_ns)

    def __getattr__(self, name):
        """转发所有未定义的方法和属性到底层子模块"""
        if self.sub_module:
            return getattr(self.sub_module, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    # 显式重写核心方法，确保接口一致性
    def post(self, *args, **kwargs):
        return self.sub_module.post(*args, **kwargs)

    def post_image(self, *args, **kwargs):
        return self.sub_module.post_image(*args, **kwargs)

    def get_vector(self, *args, **kwargs):
        return self.sub_module.get_vector(*args, **kwargs)

    def push_message(self, *args, **kwargs):
        return self.sub_module.push_message(*args, **kwargs)

    def ctx(self):
        return self.sub_module.ctx()
