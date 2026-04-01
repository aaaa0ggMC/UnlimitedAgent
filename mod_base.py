from log import log

class Module:
    shared_namespace = "_shared_mode_base"
    namespace = "mod_base"
    context : dict
    config_model = None # 用于存储 Pydantic 模型类

    @classmethod
    def get_config_model(cls):
        return cls.config_model

    def cfg(self, config: dict) -> dict:
        """
        条件控制配置加载：
        1. 优先找专属配置 instance_cfg。
        2. 如果 instance_cfg 里 "_override" 为 True，则合并 shared。
        3. 如果没有 instance_cfg，则 fallback 到 shared。
        """
        shared_cfg = config.get(self.shared_namespace, {})
        instance_cfg = config.get(self.namespace)

        # 情况 A：完全没有专属配置 -> 直接看有没有共享配置
        if instance_cfg is None:
            if shared_cfg:
                log(f"[{self.namespace}] 缺失专属配置，Fallback 降级至共享空间 \"{self.shared_namespace}\"")
                return shared_cfg
            log(f"[{self.namespace}] 警告: 专属与共享配置均为空")
            return {}

        # 情况 B：有专属配置，检查是否开启了合并开关
        # 使用 pop 将控制字段取出，避免灌入 Pydantic 时报错
        should_override = instance_cfg.pop("_override", False)

        if should_override:
            # 只有显式要求时才合并
            final_cfg = {**shared_cfg, **instance_cfg}
            log(f"[{self.namespace}] 模式: OVERRIDE (基于共享配置合并专属差异)")
            return final_cfg
        else:
            # 默认：只要有专属配置，就完全以专属为准
            log(f"[{self.namespace}] 模式: INDEPENDENT (完全使用专属配置)")
            return instance_cfg

    def ctx(self) -> dict :
        ctx = self.context.get(self.namespace)
        if ctx is None:
            self.context[self.namespace] = {}
            return self.context[self.namespace]
        return ctx

    def __init__(self,context):
        self.context = context
        pass