from log import log

class Module:
    namespace = "mod_base"
    context : dict

    def cfg(self,config : dict) -> dict :
        section = config.get(self.namespace)
        if section is None:
            log("当前配置没有对应的命名空间,Required:\"" + self.namespace + "\"")
            return {}
        else:
            return section

    def ctx(self) -> dict :
        ctx = self.context.get(self.namespace)
        if ctx is None:
            self.context[self.namespace] = {}
            return self.context[self.namespace]
        return ctx

    def __init__(self,context):
        self.context = context
        pass