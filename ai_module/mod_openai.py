import tiktoken
import base64
import os
from concurrent.futures import ThreadPoolExecutor, Future
from mod_base import Module as BaseModule
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from typing import List, overload, Union, Literal, Any, Optional
from log import log
from .base_ai_formatter import format_system_prompt

class ModuleConfig(BaseModel):
    api_key  : str
    base_url : str
    # 选择的模型
    model    : str
    # 嵌入式模型
    embedding : str
    # 系统提示词
    system   : str = Field("You are a helpful assistant")
    # 是否为流式输出
    stream_mode : bool = Field(False)
    # 在流式输出下是否回写数据到{namespace}:{stream}
    stream_write : bool = Field(False)
    raw_mode    : bool = Field(False)
    # 发送给AI的窗口大小,默认为16,小于等于0表示不考虑,包含用户和AI对话
    window_size : int = Field(16)
    # 窗口的容忍度,因此实际的窗口大小是window_size + tolerent_size
    tolerent_size : int = Field(4)
    # 超出窗口后的选择,"discard"表示抛弃(其他非法输入也是这样),"summarize"表示总结
    on_slide_window : str = Field("discard")
    # 总结用的模型
    summarize_model : str
    # 总结用的提示词
    summarize_prompt : str = Field("Summarize the chats between AI and User.")
    # 总结完毕后开始提示词
    summary_begin : str = Field("Here's the summary of previous chats:\n")
    # AI's Name
    ai_name : str = Field("assistant")
    # 用户名字
    user_name : str = Field("user")
    # 系统名字
    system_name : str = Field("system")

class Module(BaseModule):
    shared_namespace = "_shared_ai_openai"
    namespace = "ai_openai"
    config : ModuleConfig
    # 这些属于软件私下持有的数据
    client : OpenAI
    pool   : ThreadPoolExecutor
    summary_registry = None

    def __init__(self,external_config,context,pool : ThreadPoolExecutor,namespace = "ai_openai"):
        # 使用用户指定的namespace
        self.namespace = namespace

        super().__init__(context)

        self.pool = pool
        # 这里是为了截取配置的命名空间
        self.config = ModuleConfig(**self.cfg(external_config))

        # 仅在构建时格式化系统提示词
        self.config.system = format_system_prompt(self.config.system)
        self.config.summarize_prompt = format_system_prompt(self.config.summarize_prompt)

        if self.ctx().get("messages") is None:
            self.ctx()["messages"] = []
            
        # 初始化 tiktoken
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

        # 初始化OpenAI,其他的基本上也是这个套路
        self._setup_OpenAI()
        # 构建系统提示词
        if self.config.system != "":
            self.push_message(self.config.system,"system")
        log("OpenAI interface has been created.")

    @overload
    def get_vector(self, text: str, model=None, async_mode: Literal[True] = ...) -> Future: ...
    @overload
    def get_vector(self, text: str, model=None, async_mode: Literal[False] = ...) -> List[float]: ...
    @overload
    def get_vector(self, text: str, model=None, async_mode: bool = ...) -> Union[Future, List[float]]: ...
    def get_vector(self, text: str, model=None, async_mode=False):
        """
        将文本转换为向量。
        async_mode=True 时返回 Future，否则阻塞直到拿到结果。
        """
        if async_mode:
            return self.pool.submit(self._execute_vector, text, model)
        else:
            return self._execute_vector(text, model)

    def _execute_vector(self, text: str, model=None) -> List[float]:
        """
        内部执行逻辑：实际调用 OpenAI 接口
        """
        target_model = model or self.config.embedding
        
        input_text = text.replace("\n", " ")
        try:
            response = self.client.embeddings.create(
                input=input_text,
                model=target_model
            )
            return response.data[0].embedding
        except Exception as e:
            log(f"[AI] Embedding 失败: {e}")
            raise e

    def on_summary(self,fn):
        self.summary_registry = fn
    
    def post(self,message : str,model = None,stream_fn = None,trigger_push = True,is_json = False) -> Future :
        '''
        发送新的消息给AI
        '''
        return self.pool.submit(self._execute_post,message,model,stream_fn,trigger_push,is_json)

    def post_image(self, message: str, image_path: str, model=None, stream_fn=None, trigger_push=True, is_json=False) -> Future:
        '''
        发送图片消息给AI
        '''
        return self.pool.submit(self._execute_image_post, message, image_path, model, stream_fn, trigger_push, is_json)
    
    def push_message(self, content: Union[str, List[Any]], role: str = "", check_win: bool = None):
        new_entry : Any

        if role == "user":
            new_entry = {
                "role" : "user",
                "content" : content,
                "name" : self.config.user_name
            }
        elif role == "assistant":
            new_entry = {
                "role" : "assistant",
                "content" : content,
                "name" : self.config.ai_name
            }
        else:
            new_entry = {
                "role" : "system",
                "content" : format_system_prompt(content) if isinstance(content, str) else content,
                "name" : self.config.system_name
            }

        self.ctx()["messages"].append(new_entry)

        check = True if check_win is None else check_win
        if check:
            vlen = len(self.ctx()["messages"]) - self.config.window_size + self.config.tolerent_size
            if self.config.window_size >= 0 and vlen > 0:
                if self.config.on_slide_window == "summarize":
                    # 这里使用OpenAI的消息处理进行快速总结
                    self._summarize_history(self.config.tolerent_size)
                else: 
                    self.ctx()["messages"][:] = self.ctx()["messages"][-self.config.window_size: ]

    def _summarize_history(self,size):
        if len(self.ctx()["messages"]) == 0:
            return 

        system_prompt = None
        if self.ctx()["messages"][0]["role"] == "system":
            # 保留system prompt
            system_prompt = self.ctx()["messages"].pop(0)

        to_summarize = self.ctx()["messages"][:-size]
        remaining = self.ctx()["messages"][-size:]

        if self.summary_registry is not None:
            self.summary_registry(to_summarize,remaining)
        
        response = self.client.chat.completions.create(
            model=self.config.summarize_model or self.config.model,
            messages=[
                {"role": "system", "content": self.config.summarize_prompt},
                {"role": "user", "content": str(to_summarize)}
            ]
        )
        summary = response.choices[0].message.content
        msg : ChatCompletionMessageParam = {
            "role": "system",
            "content": f"Earlier conversation summary: {summary}",
            "name" : self.config.system_name
        }
        self.ctx()["messages"][:] = [msg] if system_prompt is None else [system_prompt,msg] + remaining

    def _execute_post(self,message : str,model,stream_fn,trigger_push = True,is_json = False):
        self.push_message(message,"user",False)
        
        target = self.client.chat
        if self.config.raw_mode == True:
            target = target.with_raw_response

        response = target.completions.create(
            model = self.config.model if model is None else model,
            messages = self.ctx()["messages"],
            stream = self.config.stream_mode,
            response_format = {"type" : "json_object"} if is_json else {"type" : "text"} 
        )

        ret : str
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if self.config.stream_mode:
            ret = self._handle_stream(response,stream_fn)
            # 计算 token
            prompt_content = ""
            for m in self.ctx()["messages"]:
                if isinstance(m["content"], str):
                    prompt_content += m["content"]
                elif isinstance(m["content"], list):
                    for part in m["content"]:
                        if part.get("type") == "text":
                            prompt_content += part.get("text", "")

            usage["prompt_tokens"] = len(self.encoding.encode(prompt_content))
            usage["completion_tokens"] = len(self.encoding.encode(ret))
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        else:
            ret, usage_obj = self._handle_full(response,stream_fn)
            if usage_obj:
                usage["prompt_tokens"] = usage_obj.prompt_tokens
                usage["completion_tokens"] = usage_obj.completion_tokens
                usage["total_tokens"] = usage_obj.total_tokens

        if trigger_push:
            self.push_message(ret,"assistant")
        else:
            self.ctx()["messages"].pop()
        return ret, usage

    def _execute_image_post(self, message: str, image_path: str, model: Optional[str], stream_fn: Any, trigger_push: bool = True, is_json: bool = False):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # 读取并转为 base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # 识别 mime type
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = "image/jpeg"
        if ext == ".png": mime_type = "image/png"
        elif ext == ".gif": mime_type = "image/gif"
        elif ext == ".webp": mime_type = "image/webp"

        # 构造 content parts
        content_parts = [
            {"type": "text", "text": message},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{mime_type};base64,{base64_image}"
                }
            }
        ]

        # 压入消息
        self.push_message(content_parts, "user", False)

        target = self.client.chat
        if self.config.raw_mode == True:
            target = target.with_raw_response

        # 注意：某些模型可能不支持 vision，这里依赖用户配置正确的 model (如 gpt-4o)
        response = target.completions.create(
            model = self.config.model if model is None else model,
            messages = self.ctx()["messages"],
            stream = self.config.stream_mode,
            response_format = {"type" : "json_object"} if is_json else {"type" : "text"} 
        )

        ret : str
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        if self.config.stream_mode:
            ret = self._handle_stream(response,stream_fn)
            usage["completion_tokens"] = len(self.encoding.encode(ret))
            usage["total_tokens"] = usage["completion_tokens"]
        else:
            ret, usage_obj = self._handle_full(response,stream_fn)
            if usage_obj:
                usage["prompt_tokens"] = usage_obj.prompt_tokens
                usage["completion_tokens"] = usage_obj.completion_tokens
                usage["total_tokens"] = usage_obj.total_tokens

        if trigger_push:
            self.push_message(ret, "assistant")
        else:
            self.ctx()["messages"].pop()
            
        return ret, usage

    def _handle_stream(self, response,stream_fn):
        full_content = []
        stream_iterator = response.parse() if self.config.raw_mode else response
        for chunk in stream_iterator:
            if len(chunk.choices) == 0:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                if stream_fn:
                    stream_fn(delta,self.config.stream_mode)
                full_content.append(delta)
                if self.config.stream_write:
                    self.ctx()["stream"] = "".join(full_content)
        return "".join(full_content)

    def _handle_full(self, response,stream_fn) -> tuple :
        if self.config.raw_mode:
            data = response.parse()
            self.ctx()["headers"] = dict(response.headers)
            if stream_fn:
                stream_fn(data.choices[0].message.content,self.config.stream_mode)
            return data.choices[0].message.content, data.usage
        else:
            if stream_fn:
                stream_fn(response.choices[0].message.content,self.config.stream_mode)
            return response.choices[0].message.content, response.usage

    def _setup_OpenAI(self):
        self.client = OpenAI(
            api_key = self.config.api_key,
            base_url = self.config.base_url
        )
