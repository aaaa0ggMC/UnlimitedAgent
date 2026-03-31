import datetime
import string
import re

class SafeFormatter(string.Formatter):
    r"""
    一个安全的格式化器，支持自定义格式化规范（如 [[now:%Y%m%d]]）。
    如果遇到未定义的变量，会保留原样而不是抛出异常。
    支持使用 \[ 来转义 [。
    """
    def parse(self, format_string):
        if not format_string:
            return
        # 匹配 \[ (转义) 或者 [[key:spec]]
        # 组1: 转义字符, 组2: 变量名, 组3: 格式规范
        pattern = re.compile(r'\\(.)|\[\[([^\[\]:]+)(?::([^\[\]]+))?\]\]')
        last_pos = 0
        for match in pattern.finditer(format_string):
            literal = format_string[last_pos:match.start()]
            escaped = match.group(1)
            if escaped:
                # 如果是转义字符，将其作为普通文本的一部分
                yield literal + escaped, None, None, None
            else:
                field_name = match.group(2)
                format_spec = match.group(3) or ""
                yield literal, field_name, format_spec, None
            last_pos = match.end()
        yield format_string[last_pos:], None, None, None

    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            if key in kwargs:
                return kwargs[key]
            # 返回一个占位符字符串，以便 format_field 可以识别它
            return f"__MISSING_{key}__"
        return super().get_value(key, args, kwargs)

    def format_field(self, value, format_spec):
        if isinstance(value, str) and value.startswith("__MISSING_") and value.endswith("__"):
            key = value[10:-2]
            if format_spec:
                return "[[" + f"{key}:{format_spec}" + "]]"
            return "[[" + key + "]]"
        return super().format_field(value, format_spec)

def format_system_prompt(prompt: str, extra_params: dict = None) -> str:
    r"""
    格式化系统提示词，自动注入时间等信息。
    支持的变量语法为 [[变量名]] 或 [[变量名:格式]]。
    可以使用 \[ 来转义 [。
    支持的变量：
    - [[time]]: 当前时间 (HH:MM:SS)
    - [[date]]: 当前日期 (YYYY-MM-DD)
    - [[weekday]]: 当前星期几
    - [[datetime]]: 当前日期时间 (YYYY-MM-DD HH:MM:SS)
    - [[now]]: 当前 datetime 对象，支持自定义格式化，例如 [[now:%Y%m%d]]
    - [[today]]: [[now]] 的别名
    """
    now = datetime.datetime.now()
    params = {
        "now": now,
        "today": now,
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": now.strftime("%A"),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if extra_params:
        params.update(extra_params)
    
    formatter = SafeFormatter()
    return formatter.format(prompt, **params)
