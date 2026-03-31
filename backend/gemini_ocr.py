import os
import sys
import json
import pyperclip
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到 sys.path 以加载模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_module import mod_gemini
from log import set_log_fn, log

def notify(title, message):
    """发送系统通知"""
    try:
        subprocess.run(["notify-send", title, message], check=False)
    except Exception as e:
        log(f"发送通知失败: {e}")

def clean_ocr_result(text: str) -> str:
    """
    清理 OCR 结果：
    1. 如果包含代码块 ```...```，提取其中的内容。
    2. 如果没有代码块，尝试去除常见的 AI 废话开头。
    3. 去除首尾空白。
    """
    text = text.strip()
    # 尝试匹配 ``` 及其中的内容，取第一个匹配项
    code_block_match = re.search(r"```(?:\w+)?\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 如果没有代码块，尝试去除常见的引导语
    noise_patterns = [
        r"^Here is the (?:recognized )?text:?\s*",
        r"^The (?:recognized )?text is:?\s*",
        r"^OCR result:?\s*",
        r"^If there is (?:no text|nothing), output (?:nothing|\"\")\.?\s*",
        r"^Sure, here is the text:?\s*",
        r"^OCR the text in this image\.?\s*",
        r"^DO NOT (?:explain|chat)\.?\s*",
        r"^If there is text, output IT ONLY\.?\s*",
        r"^If there are multiple lines, keep the format\.?\s*",
        r"^Extract and output ONLY the text from the image\.?\s*",
        r"^Do not include any conversation, explanation, or markdown formatting\.?\s*",
        r"^Output ONLY the extracted text\.?\s*",
        r"^If there is any text, output it only\.?\s*",
        r"^If there are multiple lines, keep the format\.?\s*"
    ]

    cleaned = text
    while True:
        original_len = len(cleaned)
        for pattern in noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if len(cleaned) == original_len:
            break

    return cleaned.strip()

def main():
    set_log_fn(print)

    # 检查命令行参数（Spectacle 会传入图片路径）
    if len(sys.argv) < 2:
        log("用法: python backend/ollama_ocr.py <image_path>")
        return

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        log(f"错误: 找不到图片文件 {image_path}")
        return

    # 发送“正在解析”通知
    notify("Gemini OCR", "正在识别图片内容，请稍候...")

    # 读取统一配置
    config_path = os.path.join(project_root, "config.json")
    if not os.path.exists(config_path):
        log(f"错误: 找不到配置文件 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        external_config = json.load(f)

    # 初始化 Gemini 模块
    # 使用 "ai_gemini" 作为 namespace，如果 config.json 中没有，则尝试从 shared 或 fallback
    context = {}
    
    # 覆盖 OCR 专属的 system prompt，确保输出纯净
    ocr_system_prompt = "You are a specialized OCR tool. Extract and output ONLY the text from the image. Do not include any conversation, explanation, or markdown formatting."
    
    # 如果 config.json 里没有 ai_gemini，我们尝试寻找一个可用的配置
    # 优先检查是否有专门为 OCR 准备的配置，或者回退到共享配置
    namespace = "gemini_ocr"
    pool = ThreadPoolExecutor(max_workers=2)
    
    # 强制覆盖为 OCR 专属 prompt
    ocr_system_prompt = "You are a specialized OCR tool. Extract and output ONLY the text from the image. Do not include any conversation, explanation, or markdown formatting."
    context["system_prompt"] = ocr_system_prompt
    
    mod = mod_gemini.Module(external_config, context, pool, namespace=namespace)

    log(f"正在使用 {mod.config.model} 进行 OCR...")
    
    try:
        # 发送图片并等待结果
        prompt = "OCR the text in this image. Output ONLY the extracted text."
        future = mod.post_image(prompt, image_path)
        raw_result, usage = future.result()
        
        # 清理结果
        result = clean_ocr_result(raw_result)
        
        # 写入剪贴板
        pyperclip.copy(result)
        
        log("-" * 30)
        log("OCR 结果 (已清理):")
        log(result)
        log("-" * 30)
        
        msg = f"识别完成，已复制到剪贴板 ({usage['total_tokens']} tokens)"
        log(msg)
        notify("Gemini OCR", msg)
        
    except Exception as e:
        error_msg = f"OCR 处理失败: {e}"
        log(error_msg)
        notify("Gemini OCR 错误", error_msg)
    finally:
        pool.shutdown()

if __name__ == "__main__":
    main()
