import os
import sys
import json
import pyperclip
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# 添加项目根目录到 sys.path 以加载模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.backend_base import BackendBase, BackendConfig
from log import set_log_fn, log

def notify(title, message):
    """发送系统通知"""
    try:
        subprocess.run(["notify-send", title, message], check=False)
    except Exception as e:
        log(f"发送通知失败: {e}")

def clean_ocr_result(text: str) -> str:
    text = text.strip()
    code_block_match = re.search(r"```(?:\w+)?\n?(.*?)\n?```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

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

class OCRConfig(BackendConfig):
    ocr_prompt: str = Field("OCR the text in this image. Output ONLY the extracted text.", description="Prompt for OCR")

class OCRBackend(BackendBase):
    config_model = OCRConfig
    def __init__(self, external_config, context, pool):
        super().__init__(external_config, context, pool, namespace="gemini_ocr")
        self.cfg_obj = OCRConfig(**self.backend_cfg)

    def run(self, image_path):
        if not os.path.exists(image_path):
            log(f"错误: 找不到图片文件 {image_path}")
            return

        # 发送“正在解析”通知
        notify("OCR", "正在识别图片内容，请稍候...")

        log(f"正在使用 {self.mod.sub_module.config.model} 进行 OCR...")
        
        try:
            # 发送图片并等待结果
            prompt = self.cfg_obj.ocr_prompt
            future = self.mod.post_image(prompt, image_path)
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
            notify("OCR", msg)
            
        except Exception as e:
            error_msg = f"OCR 处理失败: {e}"
            log(error_msg)
            notify("OCR 错误", error_msg)

def main():
    set_log_fn(print)

    if len(sys.argv) < 2:
        log("用法: python backend/gemini_ocr.py <image_path>")
        return

    image_path = sys.argv[1]

    config_path = os.path.join(project_root, "config.json")
    if not os.path.exists(config_path):
        log(f"错误: 找不到配置文件 {config_path}")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        external_config = json.load(f)

    context = {}
    pool = ThreadPoolExecutor(max_workers=2)
    
    ocr = OCRBackend(external_config, context, pool)
    ocr.run(image_path)
    
    pool.shutdown()

if __name__ == "__main__":
    main()
