import time
import atexit
from concurrent.futures import ThreadPoolExecutor, Future
from mod_base import Module as BaseModule
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, overload, Union, Literal, Optional, Any
from log import log
from .base_ai_formatter import format_system_prompt
import os

class ModuleConfig(BaseModel):
    api_key: str
    # 选择的模型
    model: str = Field("gemini-1.5-flash")
    # 嵌入式模型
    embedding: str = Field("text-embedding-004")
    # 系统提示词
    system: str = Field("You are a helpful assistant")
    # 是否开启缓存 (需要 system prompt 达到一定长度, 默认 32k 左右)
    use_cache: bool = Field(False)
    # 缓存 TTL (秒)
    cache_ttl: int = Field(3600)
    # 是否为流式输出
    stream_mode: bool = Field(False)
    # 在流式输出下是否回写数据到{namespace}:{stream}
    stream_write: bool = Field(False)
    # 发送给AI的窗口大小,默认为16,小于等于0表示不考虑,包含用户和AI对话
    window_size: int = Field(16)
    # 窗口的容忍度
    tolerent_size: int = Field(4)
    # 超出窗口后的选择
    on_slide_window: str = Field("discard")
    # 总结用的模型
    summarize_model: str = Field("gemini-1.5-flash")
    # 总结用的提示词
    summarize_prompt: str = Field("Summarize the chats between AI and User.")
    # AI's Name
    ai_name: str = Field("model")
    # 用户名字
    user_name: str = Field("user")

class Module(BaseModule):
    shared_namespace = "_shared_ai_gemini"
    namespace = "ai_gemini"
    config: ModuleConfig
    client: genai.Client
    pool: ThreadPoolExecutor
    cache_name: Optional[str] = None
    cache_expire_time: float = 0

    def __init__(self, external_config, context, pool: ThreadPoolExecutor, namespace="ai_gemini"):
        self.namespace = namespace
        super().__init__(context)
        self.pool = pool
        self.config = ModuleConfig(**self.cfg(external_config))

        # 仅在构建时格式化系统提示词
        self.config.system = format_system_prompt(self.config.system)
        self.config.summarize_prompt = format_system_prompt(self.config.summarize_prompt)

        if self.ctx().get("messages") is None:
            self.ctx()["messages"] = []

        self.client = genai.Client(api_key=self.config.api_key)
        
        # 处理缓存
        if self.config.use_cache:
            self._setup_cache()
            # 注册 atexit 钩子
            atexit.register(self.close)
            
        log(f"Gemini interface has been created for model {self.config.model}.")

    def _setup_cache(self):
        """初始化或恢复缓存"""
        try:
            log("Setting up Gemini Context Cache...")
            cache = self.client.caches.create(
                model=self.config.model,
                config=types.CreateCachedContentConfig(
                    system_instruction=self.config.system,
                    ttl=f"{self.config.cache_ttl}s",
                )
            )
            self.cache_name = cache.name
            self.cache_expire_time = time.time() + self.config.cache_ttl
            log(f"Cache created: {self.cache_name}")
        except Exception as e:
            log(f"Failed to create cache: {e}. Falling back to normal system prompt.")
            self.config.use_cache = False

    def _refresh_cache_if_needed(self):
        """如果缓存快过期了就续期"""
        if not self.cache_name:
            return
        
        # 提前 5 分钟续期
        if time.time() > self.cache_expire_time - 300:
            try:
                log(f"Refreshing cache {self.cache_name}...")
                self.client.caches.update(
                    name=self.cache_name,
                    config=types.UpdateCachedContentConfig(
                        ttl=f"{self.config.cache_ttl}s"
                    )
                )
                self.cache_expire_time = time.time() + self.config.cache_ttl
            except Exception as e:
                log(f"Failed to refresh cache: {e}")

    def close(self):
        """显式关闭缓存"""
        if self.cache_name:
            try:
                log(f"Deleting cache {self.cache_name}...")
                self.client.caches.delete(name=self.cache_name)
                self.cache_name = None
            except Exception as e:
                print(f"Failed to delete cache: {e}")

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
            res = self.client.models.embed_content(
                model=target_model,
                contents=text
            )
            return res.embeddings[0].values
        except Exception as e:
            log(f"[AI] Embedding 失败: {e}")
            raise e

    def push_message(self, content: Any, role: str = "user", check_win: bool = True):
        if role == "system" and not self.config.use_cache:
             self.ctx()["system_prompt"] = format_system_prompt(content) if isinstance(content, str) else content
             return

        new_entry = {
            "role": role if role != "assistant" else "model",
            "parts": [{"text": content}] if isinstance(content, str) else content
        }

        self.ctx()["messages"].append(new_entry)

        if check_win:
            vlen = len(self.ctx()["messages"]) - self.config.window_size + self.config.tolerent_size
            if self.config.window_size >= 0 and vlen > 0:
                if self.config.on_slide_window == "summarize":
                    self._summarize_history(self.config.tolerent_size)
                else:
                    self.ctx()["messages"][:] = self.ctx()["messages"][-self.config.window_size:]

    def _summarize_history(self, size):
        if len(self.ctx()["messages"]) <= size:
            return
        
        to_summarize = self.ctx()["messages"][:-size]
        remaining = self.ctx()["messages"][-size:]
        
        try:
            response = self.client.models.generate_content(
                model=self.config.summarize_model,
                contents=f"{self.config.summarize_prompt}\n\n{str(to_summarize)}"
            )
            summary = response.text
            msg = {
                "role": "model",
                "parts": [{"text": f"Earlier conversation summary: {summary}"}]
            }
            self.ctx()["messages"][:] = [msg] + remaining
        except Exception as e:
            log(f"Summarize failed: {e}")

    def post(self, message: str, model=None, stream_fn=None, trigger_push=True, is_json=False) -> Future:
        return self.pool.submit(self._execute_post, message, model, stream_fn, trigger_push, is_json)

    def post_image(self, message: str, image_path: str, model=None, stream_fn=None, is_json=False, metadata=None) -> Future:
        return self.pool.submit(self._execute_media_post, message, image_path, "image", model, stream_fn, is_json, metadata)

    def post_audio(self, message: str, audio_path: str, model=None, stream_fn=None, is_json=False, metadata=None) -> Future:
        return self.pool.submit(self._execute_media_post, message, audio_path, "audio", model, stream_fn, is_json, metadata)

    def post_video(self, message: str, video_path: str, model=None, stream_fn=None, is_json=False, metadata=None) -> Future:
        return self.pool.submit(self._execute_media_post, message, video_path, "video", model, stream_fn, is_json, metadata)

    def _execute_post(self, message: str, model, stream_fn, trigger_push=True, is_json=False):
        self.push_message(message, "user", False)
        self._refresh_cache_if_needed()

        config_args = {}
        if self.cache_name:
            config_args["cached_content"] = self.cache_name
        else:
            config_args["system_instruction"] = self.ctx().get("system_prompt", self.config.system)

        if is_json:
            config_args["response_mime_type"] = "application/json"

        target_model = model or self.config.model
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            if self.config.stream_mode:
                response = self.client.models.generate_content_stream(
                    model=target_model,
                    contents=self.ctx()["messages"],
                    config=types.GenerateContentConfig(**config_args)
                )
                full_text = ""
                for chunk in response:
                    text = chunk.text
                    if text:
                        full_text += text
                        if stream_fn:
                            stream_fn(text, True)
                    # 最后一个 chunk 通常包含 usage
                    if chunk.usage_metadata:
                        usage["prompt_tokens"] = chunk.usage_metadata.prompt_token_count
                        usage["completion_tokens"] = chunk.usage_metadata.candidates_token_count
                        usage["total_tokens"] = chunk.usage_metadata.total_token_count
                ret = full_text
            else:
                response = self.client.models.generate_content(
                    model=target_model,
                    contents=self.ctx()["messages"],
                    config=types.GenerateContentConfig(**config_args)
                )
                ret = response.text
                if stream_fn:
                    stream_fn(ret, False)
                if response.usage_metadata:
                    usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                    usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                    usage["total_tokens"] = response.usage_metadata.total_token_count

            if trigger_push:
                self.push_message(ret, "model")
            return ret, usage
        except Exception as e:
            log(f"Gemini Post failed: {e}")
            raise e

    def _execute_media_post(self, message: str, file_path: str, media_type: str, model, stream_fn, is_json=False, metadata=None):
        try:
            log(f"Uploading {media_type} file: {file_path}")
            if not os.path.exists(file_path):
                 raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, "rb") as f:
                file_data = f.read()
            
            mime_type = "image/jpeg"
            if media_type == "audio":
                mime_type = "audio/mpeg"
            elif media_type == "video":
                mime_type = "video/mp4"
            elif media_type == "image" and file_path.endswith(".png"):
                mime_type = "image/png"

            if metadata:
                import json as json_lib
                message = f"Metadata: {json_lib.dumps(metadata, ensure_ascii=False)}\n\nUser Message: {message}"

            content_parts = [
                {"text": message},
                {"inline_data": {"data": file_data, "mime_type": mime_type}}
            ]

            self._refresh_cache_if_needed()
            
            config_args = {}
            if self.cache_name:
                config_args["cached_content"] = self.cache_name
            else:
                config_args["system_instruction"] = self.ctx().get("system_prompt", self.config.system)

            if is_json:
                config_args["response_mime_type"] = "application/json"

            target_model = model or self.config.model
            usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
            response = self.client.models.generate_content(
                model=target_model,
                contents=content_parts,
                config=types.GenerateContentConfig(**config_args)
            )
            
            ret = response.text
            if stream_fn:
                stream_fn(ret, False)
            if response.usage_metadata:
                usage["prompt_tokens"] = response.usage_metadata.prompt_token_count
                usage["completion_tokens"] = response.usage_metadata.candidates_token_count
                usage["total_tokens"] = response.usage_metadata.total_token_count
                
            return ret, usage

        except Exception as e:
            log(f"Gemini Media Post failed: {e}")
            raise e
