import json
import time
from concurrent.futures import ThreadPoolExecutor,Future
from ai_module import mod_openai
from db import postgresql
from log import set_log_fn, log 
from typing import List

def stream_display(delta: str,stream_mode : bool):
    log(delta, end="", flush=True)

def main():
    set_log_fn(print)
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            external_config = json.load(f)
    except FileNotFoundError:
        log("错误: 找不到 config.json 文件")
        return
    context = {}
    pool = ThreadPoolExecutor(max_workers=10)
    mod = mod_openai.Module(external_config, context, pool)
    db = postgresql.DB(external_config,context)
    log(f"--- 已连接到 AI ({mod.config.model}) ---")
    log("输入 'exit' 或 'quit' 退出聊天\n")

    while True:
        user_input = input("\n用户 > ").strip()
        if user_input.lower() in ['exit', 'quit']:
            break
        if not user_input:
            continue

        if user_input.lower() == "msgs":
            log(mod.ctx()["messages"])
            continue

        log(f"AI> ", end="")
        future2 = mod.get_vector(user_input,async_mode = True)
        future = mod.post(user_input, stream_fn=stream_display)
        future.result()
        print("\nRAG Size:", len(future2.result()))
        log("\n") 
    pool.shutdown()

if __name__ == "__main__":
    main()