import json
import os
import sys
import hashlib
import shutil
import mimetypes
import time
import random
import string
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field

# 添加项目根目录到 sys.path 以加载模块
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ai_module import mod_general
from db import postgresql
from backend.backend_base import BackendBase, BackendConfig
from log import set_log_fn, log
from PIL import Image
from PIL.ExifTags import TAGS
import mutagen
from pgvector import Vector

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

class AssetConfig(BackendConfig):
    source_dir: str = Field(".", description="Source directory to scan")
    build_dir: str = Field("./build", description="Build directory for assets")
    supported_extensions: list = Field(['.jpg', '.jpeg', '.png', '.mp4', '.mp3', '.wav', '.mov', '.m4a'], description="Supported file extensions")

class AssetBuilder(BackendBase):
    config_model = AssetConfig
    def __init__(self, external_config, context, pool, source_dir=None, build_dir=None):
        super().__init__(external_config, context, pool, namespace="asset_builder")
        self.cfg_obj = AssetConfig(**self.backend_cfg)
        
        # 允许从外部参数或配置中获取路径
        self.source_dir = Path(source_dir) if source_dir else Path(self.cfg_obj.source_dir)
        self.build_dir = Path(build_dir) if build_dir else Path(self.cfg_obj.build_dir)
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # embed_mod 用于嵌入 (RAG)，独立于默认的 self.mod (scanning)
        # 支持从 backend 自身配置中穿透嵌入 AI 的配置
        embed_instance_cfg = self.backend_cfg.get("ollama_embed", {})
        syn_embed_config = external_config.copy()
        syn_embed_config["ollama_embed"] = embed_instance_cfg
        self.embed_mod = mod_general.Module(syn_embed_config, context, pool, namespace="ollama_embed")
        
        # postgresql 用于存储 RAG
        # 支持从 backend 自身配置中穿透数据库配置
        db_instance_cfg = self.backend_cfg.get("db_pg", {})
        synthetic_config = external_config.copy()
        # 将子配置注入到 synthetic_config 中，使得 DB 模块能通过 self.cfg() 拿到它
        synthetic_config["db_pg"] = db_instance_cfg
        
        self.db = postgresql.DB(synthetic_config, context, namespace="db_pg")
        
        self.supported_extensions = set(self.cfg_obj.supported_extensions)

    def is_processed(self, original_path):
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
        
        file_hash = get_file_hash(file_path)
        
        # 增量检查：如果数据库中已经有这个 Hash，则跳过扫描
        existing = self.db.get("vec_qwen3_4b", pks=[file_hash])
        if existing:
            log(f"Skipping {file_path.name}: Already analyzed and indexed (Hash: {file_hash[:8]})")
            return

        # 兜底检查：本地 JSON 是否已存在
        if self.is_processed(file_path):
            log(f"Skipping {file_path.name} (Sidecar JSON already exists)")
            return

        log(f"Processing: {file_path.name} ...")
        
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
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
                "local_metadata": local_metadata,
                "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "usage": usage
            }
            
            json_path = target_path.with_suffix(target_path.suffix + ".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_data, f, ensure_ascii=False, indent=2)
                
            log(f"Successfully built: {pref_name}")
            
        except Exception as e:
            log(f"Error processing {file_path.name}: {e}")

    def scan(self):
        log(f"Starting scan in {self.source_dir} ...")
        files = []
        for root, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                files.append(Path(root) / filename)
        
        log(f"Found {len(files)} files. Starting build pipeline...")
        for file in files:
            self.process_file(file)
            
        log("Asset scanning and generation completed.")

    def build_rag(self):
        log(f"Starting RAG build from {self.build_dir} ...")
        json_files = list(self.build_dir.glob("*.json"))
        
        if not json_files:
            log("No JSON files found in build directory.")
            return

        for json_path in json_files:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                file_hash = data.get("file_hash")
                scan_time = data.get("scan_time", "")
                if not file_hash:
                    log(f"Skipping {json_path.name}: No file_hash found.")
                    continue

                # 增量检查：如果数据库中 Hash 和 ScanTime 都匹配，则跳过 Embedding
                existing = self.db.get("vec_qwen3_4b", pks=[file_hash])
                if existing:
                    db_scan_time = existing[0].get("data", {}).get("scan_time", "")
                    if db_scan_time == scan_time:
                        log(f"Skipping Embedding: {json_path.name} (Up to date)")
                        continue

                # 构建用于嵌入的文本
                text_content = f"Description: {data.get('description', '')}. "
                text_content += f"Tags: {', '.join(data.get('tags', []))}. "
                text_content += f"Original Name: {data.get('original_name', '')}"
                
                log(f"Embedding: {json_path.name} (Hash: {file_hash[:8]}) ...")
                vector = self.embed_mod.get_vector(text_content)
                
                # 存入数据库，使用 file_hash 作为 pk 防止重复
                self.db.store("vec_qwen3_4b", {
                    "file_hash": file_hash,
                    "vector": vector,
                    "data": {
                        "filename": json_path.name, # 只存文件名
                        "original_name": data.get("original_name", ""),
                        "prefered_name": json_path.stem,
                        "description": data.get("description", ""),
                        "tags": data.get("tags", []),
                        "scan_time": scan_time
                    }
                }, pk="file_hash", try_vectorize=True)
                
            except Exception as e:
                log(f"Error building RAG for {json_path.name}: {e}")

        log("RAG build completed.")

    def search_rag(self, query, count=5):
        log(f"Searching RAG for: '{query}' (top {count}) ...")
        try:
            query_vector = self.embed_mod.get_vector(query)
            results = self.db.find_vectors("vec_qwen3_4b", Vector(query_vector), limit=count, norm_val=3)
            
            if not results:
                log("No similar assets found.")
                return

            log(f"\nFound {len(results)} matches:")
            for i, res in enumerate(results, 1):
                # 转换余弦距离为余弦相似度 (1 - distance)
                dist = res.get("dist", 1.0)
                similarity = 1.0 - dist
                asset_data = res.get("data", {})
                log(f"{i}. {asset_data.get('original_name')} (Similarity: {similarity:.4f})")
                log(f"   Filename: {asset_data.get('filename')}")
                log(f"   Description: {asset_data.get('description')}")
                log("-" * 20)
                
        except Exception as e:
            log(f"Search failed: {e}")

if __name__ == "__main__":
    import argparse
    set_log_fn(print)
    
    config_path = os.path.join(project_root, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found at {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        external_config = json.load(f)

    parser = argparse.ArgumentParser(description="Asset Builder RAG Tool")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    scan_parser = subparsers.add_parser("scan", help="Scan a directory for media and analyze")
    scan_parser.add_argument("source_dir", help="Directory to scan")
    scan_parser.add_argument("--build_dir", help="Directory to store JSON sidecars and copies")

    build_parser = subparsers.add_parser("build", help="Index analyzed JSON files into Postgresql RAG")
    build_parser.add_argument("--build_dir", help="Directory where JSON sidecars are stored")
    
    search_parser = subparsers.add_parser("search", help="Search for assets using RAG")
    search_parser.add_argument("count", type=int, help="Number of results to return")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("--build_dir", help="Build directory for context")

    args = parser.parse_args()
    context = {}
    pool = ThreadPoolExecutor(max_workers=5)
    
    if args.command == "scan":
        builder = AssetBuilder(external_config, context, pool, source_dir=args.source_dir, build_dir=args.build_dir)
        builder.scan()
    elif args.command == "build":
        builder = AssetBuilder(external_config, context, pool, build_dir=args.build_dir)
        builder.build_rag()
    elif args.command == "search":
        builder = AssetBuilder(external_config, context, pool, build_dir=args.build_dir)
        builder.search_rag(args.query, args.count)
    else:
        parser.print_help()
    
    pool.shutdown()
