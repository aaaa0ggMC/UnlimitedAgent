import json
import os
import hashlib
import shutil
import mimetypes
import time
import random
import string
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from ai_module import mod_gemini
from log import set_log_fn, log
from PIL import Image
from PIL.ExifTags import TAGS
import mutagen

def get_file_hash(file_path):
    """计算文件的 SHA256 哈希值"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def generate_random_hash(length=4):
    """生成随机哈希字符串用于文件名防止碰撞"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def make_serializable(obj):
    """递归将对象转换为 JSON 可序列化的格式"""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [make_serializable(v) for v in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8', errors='replace')
        except:
            return str(obj)
    # 处理 Pillow 的 IFDRational 等特殊类型
    if hasattr(obj, 'numerator') and hasattr(obj, 'denominator'):
        if obj.denominator == 0:
            return 0
        return float(obj.numerator) / obj.denominator
    return str(obj)

def extract_metadata(file_path):
    """提取媒体文件的元数据 (EXIF, Mutagen)"""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    metadata = {}
    try:
        if suffix in ['.jpg', '.jpeg', '.png']:
            with Image.open(file_path) as img:
                # 获取基本信息
                metadata['format'] = img.format
                metadata['width'], metadata['height'] = img.size
                metadata['mode'] = img.mode
                
                # 提取 EXIF
                exif_data = img._getexif()
                if exif_data:
                    exif = {}
                    for tag, value in exif_data.items():
                        decoded = TAGS.get(tag, tag)
                        exif[decoded] = value
                    metadata['exif'] = exif
                    
        elif suffix in ['.mp3', '.wav', '.mp4', '.mov', '.m4a']:
            audio = mutagen.File(file_path)
            if audio:
                tags = {}
                for key in audio.keys():
                    tags[key] = audio[key]
                metadata['tags'] = tags
                
                if audio.info:
                    info = {
                        'length': getattr(audio.info, 'length', 0),
                        'bitrate': getattr(audio.info, 'bitrate', 0),
                        'sample_rate': getattr(audio.info, 'sample_rate', 0),
                        'channels': getattr(audio.info, 'channels', 0)
                    }
                    metadata['info'] = info
    except Exception as e:
        log(f"Metadata extraction failed for {file_path.name}: {e}")
    
    return make_serializable(metadata)

class AssetBuilder:
    def __init__(self, source_dir, build_dir, config_path="config.json"):
        self.source_dir = Path(source_dir)
        self.build_dir = Path(build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.external_config = json.load(f)
            
        self.context = {}
        self.pool = ThreadPoolExecutor(max_workers=5)
        # mod_gemini 会自动根据 namespace 读取 config 中的配置
        self.mod = mod_gemini.Module(self.external_config, self.context, self.pool, namespace="asset_builder")
        
        # 强制非流式
        self.mod.config.stream_mode = False
        self.supported_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.mp3', '.wav', '.mov', '.m4a'}

    def is_processed(self, original_path):
        """检查是否已经处理过"""
        if not hasattr(self, '_processed_cache'):
            self._processed_cache = set()
            for json_file in self.build_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "original_name" in data:
                            self._processed_cache.add(data["original_name"])
                except:
                    continue
        return original_path.name in self._processed_cache

    def process_file(self, file_path):
        if file_path.suffix.lower() not in self.supported_extensions:
            return
        
        if self.is_processed(file_path):
            log(f"Skipping {file_path.name} (already processed)")
            return

        log(f"Processing: {file_path.name} ...")
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_hash = get_file_hash(file_path)
        
        # 提取本地元数据
        local_metadata = extract_metadata(file_path)
        
        # 发送给 AI 的元数据，包含 original_filename
        ai_metadata = {**local_metadata, "original_filename": file_path.name}
        
        # 这里使用简单的消息，主要的指令逻辑应放在 namespace 的 system prompt 中
        ai_msg = "Please analyze this asset."
        
        try:
            if mime_type.startswith("image/"):
                future = self.mod.post_image(ai_msg, str(file_path), is_json=True, metadata=ai_metadata)
            elif mime_type.startswith("audio/"):
                future = self.mod.post_audio(ai_msg, str(file_path), is_json=True, metadata=ai_metadata)
            elif mime_type.startswith("video/"):
                future = self.mod.post_video(ai_msg, str(file_path), is_json=True, metadata=ai_metadata)
            else:
                log(f"Unsupported MIME type: {mime_type} for {file_path}")
                return

            res_text, usage = future.result()
            ai_data = json.loads(res_text)
            
            # 准备物理重命名
            pref_name = ai_data.get("prefered_name", f"unnamed_{generate_random_hash()}.{file_path.suffix}")
            
            # 检查碰撞
            target_path = self.build_dir / pref_name
            while target_path.exists():
                name_parts = pref_name.rsplit('.', 1)
                pref_name = f"{name_parts[0]}_{generate_random_hash()}.{name_parts[1]}"
                target_path = self.build_dir / pref_name

            # 复制文件
            shutil.copy2(file_path, target_path)
            
            # 写入 Sidecar JSON
            sidecar_data = {
                **ai_data,
                "original_name": file_path.name,
                "original_path": str(file_path.absolute()),
                "file_hash": file_hash,
                "local_metadata": local_metadata, # 此处 local_metadata 不包含重复的 original_filename
                "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "usage": usage
            }
            
            json_path = target_path.with_suffix(target_path.suffix + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, ensure_ascii=False, indent=2)
                
            log(f"Successfully built: {pref_name}")
            
        except Exception as e:
            log(f"Error processing {file_path.name}: {e}")

    def run(self):
        log(f"Starting scan in {self.source_dir} ...")
        files = []
        for root, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                files.append(Path(root) / filename)
        
        log(f"Found {len(files)} files. Starting build pipeline...")
        for file in files:
            self.process_file(file)
            
        self.pool.shutdown()
        log("Asset building completed.")

if __name__ == "__main__":
    import sys
    set_log_fn(print)
    
    if len(sys.argv) < 2:
        print("Usage: python asset_builder.py <source_directory>")
        sys.exit(1)
        
    source_path = sys.argv[1]
    if not os.path.exists(source_path):
        print(f"Error: Source directory '{source_path}' does not exist.")
        sys.exit(1)

    try:
        builder = AssetBuilder(source_path, "./build")
        builder.run()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
