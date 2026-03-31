import json
import os
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from ai_module import mod_gemini
from log import set_log_fn, log

# For Tab-Completion
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.styles import Style

def stream_display(delta: str, stream_mode: bool):
    print(delta, end="", flush=True)

def estimate_cost(usage, model="gemini-1.5-flash"):
    # Gemini 1.5 Flash 费率 (128k context)
    prompt_cost = (usage["prompt_tokens"] / 1_000_000) * 0.075
    completion_cost = (usage["completion_tokens"] / 1_000_000) * 0.3
    return prompt_cost + completion_cost

def main():
    set_log_fn(print)
    
    config_path = "config.json"
    if not os.path.exists(config_path):
        config_path = "config.example.json"
        log(f"Warning: config.json not found, using {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            external_config = json.load(f)
    except Exception as e:
        log(f"Error loading config: {e}")
        return

    context = {}
    pool = ThreadPoolExecutor(max_workers=5)
    
    # 初始化 Gemini 模块
    mod = mod_gemini.Module(external_config, context, pool)
    # 强制开启流式模式
    mod.config.stream_mode = True

    # 路径补全器，排除常见的无关目录
    completer = PathCompleter(
        only_directories=False,
        expanduser=True
    )
    
    # 定义提示符样式
    style = Style.from_dict({
        'prompt': '#00ff00 bold',
    })
    
    session = PromptSession(completer=completer, style=style)

    log("\n--- Gemini File Scanner & Summarizer (Tab-Completion Enabled) ---")
    log("Type a file path (Press Tab to autocomplete), or 'exit' to quit.\n")

    while True:
        try:
            # 使用 prompt_toolkit 进行交互
            file_input = session.prompt("File Path > ").strip()
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

        prompt = "Please summarize the content of this file in detail."
        future = None

        if mime_type:
            if mime_type.startswith("image/"):
                future = mod.post_image(prompt, file_input, stream_fn=stream_display)
            elif mime_type.startswith("audio/"):
                future = mod.post_audio(prompt, file_input, stream_fn=stream_display)
            elif mime_type.startswith("video/"):
                future = mod.post_video(prompt, file_input, stream_fn=stream_display)
            else:
                try:
                    with open(file_input, "r", encoding="utf-8") as f:
                        text_content = f.read()
                    future = mod.post(f"{prompt}\n\nContent:\n{text_content}", stream_fn=stream_display)
                except Exception as e:
                    log(f"Error reading file as text: {e}")
                    continue
        else:
            log("Unknown file type, trying as text...")
            try:
                with open(file_input, "r", encoding="utf-8") as f:
                    text_content = f.read()
                future = mod.post(f"{prompt}\n\nContent:\n{text_content}", stream_fn=stream_display)
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
                cost = estimate_cost(usage)
                log(f"Estimated Cost: ${cost:.6f}")
                log("-" * 30)
            except Exception as e:
                log(f"Error during processing: {e}")

    pool.shutdown()

if __name__ == "__main__":
    main()
