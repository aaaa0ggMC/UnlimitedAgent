import sys

def _default_log_fn(data, *args, **kwargs):
    pass

_module = sys.modules[__name__]

_module._log_fn = _default_log_fn

def log(data, *args, **kwargs):
    try:
        if args and isinstance(data, str):
            try:
                data = data.format(*args)
            except:
                pass
        _module._log_fn(data, *args, **kwargs)
    except Exception:
        pass

def set_log_fn(fn):
    if not callable(fn):
        raise TypeError("日志函数必须是可调用的")
    
    _module._log_fn = fn
