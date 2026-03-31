import os
import base64
from concurrent.futures import ThreadPoolExecutor, Future
from mod_base import Module as BaseModule
from ollama import Client
from pydantic import BaseModel, Field
from typing import List, overload, Union, Literal, Optional, Any
from log import log
from .base_ai_formatter import format_system_prompt

class ModuleConfig(BaseModel):
    # Ollama 的服务地址，默认连接本地
    base_url: str = Field("http://localhost:11434")
    # 选择的模型
    model: str = Field("llama3")
    # 嵌入式模型
    embedding: str = Field("llama3")
    # 系统提示词
    system: str = Field("You are a helpful assistant")
    # 模型保活时间，默认足够长 (例如 15 分钟) 用于自动卸载 & 刷新模型
    # 可以是字符串如 "5m", "15m", "1h" 或者秒数
    keep_alive: str = Field("15m")
    # 是否为流式输出
    stream_mode: bool = Field(False)
    # 在流式输出下是否回写数据到 {namespace}:{stream}
    stream_write: bool = Field(False)
    # 发送给 AI 的窗口大小, 默认为 16, 小于等于 0 表示不考虑, 包含用户和 AI 对话
    window_size: int = Field(16)
    # 窗口的容忍度
    tolerent_size: int = Field(4)
    # 超出窗口后的选择 ("discard" 或 "summarize")
    on_slide_window: str = Field("discard")
    # 总结用的模型
    summarize_model: str = Field("llama3")
    # 总结用的提示词
    summarize_prompt: str = Field("Summarize the chats between AI and User.")
    # AI's Name
    ai_name: str = Field("assistant")
    # 用户名字
    user_name: str = Field("user")

class Module(BaseModule):
    shared_namespace = "_shared_ai_ollama"
    namespace = "ai_ollama"
    config: ModuleConfig
    client: Client
    pool: ThreadPoolExecutor

    def __init__(self, external_config, context, pool: ThreadPoolExecutor, namespace="ai_ollama"):
        self.namespace = namespace
        super().__init__(context)
        self.pool = pool
        self.config = ModuleConfig(**self.cfg(external_config))

        # 仅在构建时格式化系统提示词
        self.config.system = format_system_prompt(self.config.system)
        self.config.summarize_prompt = format_system_prompt(self.config.summarize_prompt)

        if self.ctx().get("messages") is None:
            self.ctx()["messages"] = []

        # 初始化 Ollama 客户端
        self.client = Client(host=self.config.base_url)
        
        # 尝试将系统提示词作为第一条消息推入
        if self.config.system:
            # 对于 Ollama，系统提示词通常作为 role: 'system' 的消息
            # 如果已有消息则不再重复推送（除非是全新的上下文）
            if not self.ctx()["messages"]:
                self.push_message(self.config.system, role="system", check_win=False)
            
        log(f"Ollama interface has been created for model {self.config.model} at {self.config.base_url}.")

    @overload
    def get_vector(self, text: str, model=None, async_mode: Literal[True] = ...) -> Future: ...
    @overload
    def get_vector(self, text: str, model=None, async_mode: Literal[False] = ...) -> List[float]: ...
    def get_vector(self, text: str, model=None, async_mode=False):
        if async_mode:
            return self.pool.submit(self._execute_vector, text, model)
        else:
            return self._execute_vector(text, model)

    def _execute_vector(self, text: str, model=None) -> List[float]:
        target_model = model or self.config.embedding
        try:
            res = self.client.embeddings(
                model=target_model,
                prompt=text,
                keep_alive=self.config.keep_alive
            )
            return res.embedding
        except Exception as e:
            log(f"[AI] Ollama Embedding 失败: {e}")
            raise e

    def push_message(self, content: Any, role: str = "user", images: List[bytes] = None, check_win: bool = True):
        # 转换角色名以符合规范 (user, assistant, system)
        role_map = {
            "user": "user",
            "assistant": "assistant",
            "model": "assistant",
            "system": "system"
        }
        target_role = role_map.get(role, role)

        new_entry = {
            "role": target_role,
            "content": content
        }
        if images:
            new_entry["images"] = images

        self.ctx()["messages"].append(new_entry)

        if check_win:
            vlen = len(self.ctx()["messages"]) - self.config.window_size + self.config.tolerent_size
            if self.config.window_size >= 0 and vlen > 0:
                if self.config.on_slide_window == "summarize":
                    self._summarize_history(self.config.tolerent_size)
                else:
                    # 尽可能保留系统提示词（如果有）
                    system_msgs = [m for m in self.ctx()["messages"] if m["role"] == "system"]
                    other_msgs = [m for m in self.ctx()["messages"] if m["role"] != "system"]
                    self.ctx()["messages"][:] = system_msgs + other_msgs[-self.config.window_size:]

    def _summarize_history(self, size):
        if len(self.ctx()["messages"]) <= size:
            return
        
        # 区分系统提示词和对话历史
        system_msgs = [m for m in self.ctx()["messages"] if m["role"] == "system"]
        to_summarize = [m for m in self.ctx()["messages"] if m["role"] != "system"][:-size]
        remaining = [m for m in self.ctx()["messages"] if m["role"] != "system"][-size:]
        
        try:
            # 总结时暂时不包括图片，因为图片可能非常大且大多数总结模型不需要图片
            # 我们只总结文本内容
            text_only_history = []
            for m in to_summarize:
                text_only_history.append({"role": m["role"], "content": m["content"]})

            response = self.client.chat(
                model=self.config.summarize_model,
                messages=[
                    {"role": "system", "content": self.config.summarize_prompt},
                    {"role": "user", "content": f"Summarize these messages: {str(text_only_history)}"}
                ],
                keep_alive=self.config.keep_alive
            )
            summary = response.message.content
            msg = {
                "role": "system",
                "content": f"Earlier conversation summary: {summary}"
            }
            self.ctx()["messages"][:] = system_msgs + [msg] + remaining
        except Exception as e:
            log(f"Ollama Summarize failed: {e}")

    def post(self, message: str, model=None, stream_fn=None, trigger_push=True, is_json=False) -> Future:
        return self.pool.submit(self._execute_post, message, model, stream_fn, trigger_push, is_json)

    def post_image(self, message: str, image_path: str, model=None, stream_fn=None, trigger_push=True, is_json=False) -> Future:
        return self.pool.submit(self._execute_image_post, message, image_path, model, stream_fn, trigger_push, is_json)

    def _execute_post(self, message: str, model, stream_fn, trigger_push=True, is_json=False):
        self.push_message(message, "user", check_win=False)
        
        target_model = model or self.config.model
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            if self.config.stream_mode:
                response = self.client.chat(
                    model=target_model,
                    messages=self.ctx()["messages"],
                    stream=True,
                    format="json" if is_json else "",
                    keep_alive=self.config.keep_alive
                )
                full_text = ""
                for chunk in response:
                    text = chunk.message.content
                    if text:
                        full_text += text
                        if stream_fn:
                            stream_fn(text, True)
                    
                    if chunk.done:
                        usage["prompt_tokens"] = getattr(chunk, 'prompt_eval_count', 0) or 0
                        usage["completion_tokens"] = getattr(chunk, 'eval_count', 0) or 0
                        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                ret = full_text
            else:
                response = self.client.chat(
                    model=target_model,
                    messages=self.ctx()["messages"],
                    stream=False,
                    format="json" if is_json else "",
                    keep_alive=self.config.keep_alive
                )
                ret = response.message.content
                if stream_fn:
                    stream_fn(ret, False)
                
                usage["prompt_tokens"] = getattr(response, 'prompt_eval_count', 0) or 0
                usage["completion_tokens"] = getattr(response, 'eval_count', 0) or 0
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            if trigger_push:
                self.push_message(ret, "assistant")
            return ret, usage
        except Exception as e:
            log(f"Ollama Post failed: {e}")
            raise e

    def _execute_image_post(self, message: str, image_path: str, model, stream_fn, trigger_push=True, is_json=False):
        try:
            if not os.path.exists(image_path):
                 raise FileNotFoundError(f"Image not found: {image_path}")
            
            with open(image_path, "rb") as f:
                image_data = f.read()
            
            # Ollama 的 chat 接口可以直接接收 images 列表（base64 或 bytes）
            # 库通常处理 bytes 为 base64
            
            self.push_message(message, "user", images=[image_data], check_win=False)

            target_model = model or self.config.model
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

            if self.config.stream_mode:
                response = self.client.chat(
                    model=target_model,
                    messages=self.ctx()["messages"],
                    stream=True,
                    format="json" if is_json else "",
                    keep_alive=self.config.keep_alive
                )
                full_text = ""
                for chunk in response:
                    text = chunk.message.content
                    if text:
                        full_text += text
                        if stream_fn:
                            stream_fn(text, True)
                    if chunk.done:
                        usage["prompt_tokens"] = getattr(chunk, 'prompt_eval_count', 0) or 0
                        usage["completion_tokens"] = getattr(chunk, 'eval_count', 0) or 0
                        usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                ret = full_text
            else:
                response = self.client.chat(
                    model=target_model,
                    messages=self.ctx()["messages"],
                    stream=False,
                    format="json" if is_json else "",
                    keep_alive=self.config.keep_alive
                )
                ret = response.message.content
                if stream_fn:
                    stream_fn(ret, False)
                
                usage["prompt_tokens"] = getattr(response, 'prompt_eval_count', 0) or 0
                usage["completion_tokens"] = getattr(response, 'eval_count', 0) or 0
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

            if trigger_push:
                self.push_message(ret, "assistant")
            
            return ret, usage

        except Exception as e:
            log(f"Ollama Image Post failed: {e}")
            raise e
