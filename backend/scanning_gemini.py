import json
import os
import sys
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 添加项目根目录到 sys.path 以加载模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pydantic import BaseModel, Field
from backend.backend_base import BackendBase, BackendConfig
from log import set_log_fn, log

# For Tab-Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.styles import Style

def stream_display(delta: str, stream_mode: bool):
    print(delta, end="", flush=True)

def estimate_cost(usage, model="gemini-1.5-flash"):
    prompt_cost = (usage["prompt_tokens"] / 1_000_000) * 0.075
    completion_cost = (usage["completion_tokens"] / 1_000_000) * 0.3
    return prompt_cost + completion_cost

class ScannerConfig(BackendConfig):
    summary_prompt: str = Field("Please summarize the content of this file in detail.", description="Prompt for summarizing files")

class ScannerBackend(BackendBase):
    config_model = ScannerConfig
    def __init__(self, external_config, context, pool):
        super().__init__(external_config, context, pool, namespace="scanning_gemini")
        self.cfg_obj = ScannerConfig(**self.backend_cfg)
        
        # 路径补全器，排除常见的无关目录
        self.completer = PathCompleter(
            only_directories=False,
            expanduser=True
        )
        
        # 定义提示符样式
        self.style = Style.from_dict({
            'prompt': '#00ff00 bold',
        })
        
        self.session = PromptSession(completer=self.completer, style=self.style)

    def run(self):
        log(f"\n--- AI File Scanner & Summarizer ({self.mod.sub_module.config.model}) ---")
        log("Type a file path (Press Tab to autocomplete), or 'exit' to quit.\n")

        while True:
            try:
                file_input = self.session.prompt("File Path > ").strip()
            except KeyboardInterrupt:
                continue
            except EOFError:
                break

            if file_input.lower() in ['exit', 'quit']:
                break
            if not file_input:
                continue

            if not os.path.exists(file_input):
                log(f"Error: File '{file_input}' does not exist.")
                continue

            if os.path.isdir(file_input):
                log(f"'{file_input}' is a directory. Please provide a file path.")
                continue

            mime_type, _ = mimetypes.guess_type(file_input)
            log(f"Detected MIME type: {mime_type}")

            prompt = self.get_cfg("summary_prompt", "Please summarize the content of this file in detail.")
            future = None

            if mime_type:
                if mime_type.startswith("image/"):
                    future = self.mod.post_image(prompt, file_input, stream_fn=stream_display)
                elif mime_type.startswith("audio/"):
                    future = self.mod.post_audio(prompt, file_input, stream_fn=stream_display)
                elif mime_type.startswith("video/"):
                    future = self.mod.post_video(prompt, file_input, stream_fn=stream_display)
                else:
                    try:
                        with open(file_input, "r", encoding="utf-8") as f:
                            text_content = f.read()
                        future = self.mod.post(f"{prompt}\n\nContent:\n{text_content}", stream_fn=stream_display)
                    except Exception as e:
                        log(f"Error reading file as text: {e}")
                        continue
            else:
                log("Unknown file type, trying as text...")
                try:
                    with open(file_input, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    future = self.mod.post(f"{prompt}\n\nContent:\n{text_content}", stream_fn=stream_display)
                except Exception as e:
                    log(f"Error: {e}")
                    continue

            if future:
                log("\n\n--- Summary ---")
                try:
                    ret, usage = future.result()
                    log(f"\n\n--- Usage Stats ---")
                    log(f"Prompt Tokens: {usage['prompt_tokens']}")
                    log(f"Completion Tokens: {usage['completion_tokens']}")
                    log(f"Total Tokens: {usage['total_tokens']}")
                    cost = estimate_cost(usage, self.mod.sub_module.config.model)
                    log(f"Estimated Cost: ${cost:.6f}")
                    log("-" * 30)
                except Exception as e:
                    log(f"Error during processing: {e}")

def main():
    set_log_fn(print)
    
    config_path = os.path.join(project_root, "config.json")
    if not os.path.exists(config_path):
        log(f"Error: config.json not found at {config_path}")
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            external_config = json.load(f)
    except Exception as e:
        log(f"Error loading config: {e}")
        return

    context = {}
    pool = ThreadPoolExecutor(max_workers=5)
    
    scanner = ScannerBackend(external_config, context, pool)
    scanner.run()
    
    pool.shutdown()

if __name__ == "__main__":
    main()
