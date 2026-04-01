import os
import json
import sys
import requests
import time
from pathlib import Path
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor

# 确保能找到根目录
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.backend_base import BackendBase, BackendConfig
from ai_module import mod_general
from db import postgresql
from log import log

# 常量设置
NCM_BASE_URL = "http://localhost:3000" 

class AIDJConfig(BackendConfig):
    source_dir: str = Field(".", description="Music directory to scan")
    ncm_api_url: str = Field("http://localhost:3000", description="NCM API base URL")
    supported_extensions: list = Field(['.mp3', '.flac', '.wav', '.m4a'], description="Music file extensions")

class AIDJRag(BackendBase):
    config_model = AIDJConfig

    def __init__(self, external_config, context, pool, source_dir=None):
        super().__init__(external_config, context, pool, namespace="aidj_rag")
        self.cfg_obj = AIDJConfig(**self.backend_cfg)
        self.source_dir = Path(source_dir) if source_dir else Path(self.cfg_obj.source_dir)
        
        # 使用数据库
        db_instance_cfg = self.backend_cfg.get("db_pg", {})
        syn_db_cfg = external_config.copy()
        syn_db_cfg["db_pg"] = db_instance_cfg
        self.db = postgresql.DB(syn_db_cfg, context, namespace="db_pg")

        # 使用嵌入 AI (复用 ollama_embed 逻辑)
        embed_instance_cfg = self.backend_cfg.get("ollama_embed", {})
        syn_embed_config = external_config.copy()
        syn_embed_config["ollama_embed"] = embed_instance_cfg
        self.embed_mod = mod_general.Module(syn_embed_config, context, pool, namespace="ollama_embed")

        self.supported_extensions = set(self.cfg_obj.supported_extensions)

    def _get_song_ai_info(self, song_name, lyrics):
        """调用 AI 提取歌曲元数据"""
        prompt = f"请根据以下歌曲信息提取 JSON (language, emotion, genre, loudness, review)。歌曲名: {song_name}, 歌词片段: {lyrics[:500]}"
        try:
            future = self.mod.post(prompt, is_json=True)
            res_text, _ = future.result()
            return json.loads(res_text)
        except Exception as e:
            log(f"AI analysis failed for {song_name}: {e}")
            return None

    def _sync_song(self, file_path):
        """同步单首歌曲"""
        song_name = file_path.stem
        
        # 1. 查重
        existing = self.db.get("music_rag", pks=[song_name])
        if existing:
            log(f"Skipping: {song_name} (Already in RAG)")
            return

        log(f"Syncing: {song_name} ...")
        
        try:
            # 2. 从 NCM 获取歌词
            search_res = requests.get(f"{self.cfg_obj.ncm_api_url}/search?keywords={song_name}&limit=1", timeout=5).json()
            if search_res.get('code') != 200 or not search_res.get('result', {}).get('songs'):
                log(f"NCM Search failed for {song_name}")
                return
            
            sid = search_res['result']['songs'][0]['id']
            lyric_res = requests.get(f"{self.cfg_obj.ncm_api_url}/lyric?id={sid}", timeout=5).json()
            raw_lyric = lyric_res.get('lrc', {}).get('lyric', "暂无歌词")

            # 3. AI 提取元数据
            ai_info = self._get_song_ai_info(song_name, raw_lyric)
            if not ai_info: return

            # 4. 构建 Embedding 文本
            embed_text = f"Title: {song_name}. Genre: {ai_info.get('genre')}. Emotion: {ai_info.get('emotion')}. Lyrics: {raw_lyric[:300]}"
            vector = self.embed_mod.get_vector(embed_text)

            # 5. 存储到数据库
            self.db.store("music_rag", {
                "title": song_name,
                "vector": vector,
                "data": {
                    **ai_info,
                    "file_path": str(file_path.absolute()),
                    "sync_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, pk="title", try_vectorize=True)
            
            log(f"Successfully indexed: {song_name}")

        except Exception as e:
            log(f"Failed to sync {song_name}: {e}")

    def scan(self):
        log(f"Scanning music in {self.source_dir} ...")
        files = []
        for root, _, filenames in os.walk(self.source_dir):
            for filename in filenames:
                fp = Path(root) / filename
                if fp.suffix.lower() in self.supported_extensions:
                    files.append(fp)
        
        log(f"Found {len(files)} songs. Starting sync...")
        for file in files:
            self._sync_song(file)
        log("AIDJ RAG sync completed.")

    def search(self, query, count=5):
        log(f"Searching music for: '{query}' ...")
        try:
            query_vector = self.embed_mod.get_vector(query)
            # PostgreSQL pgvector 搜索，norm_val=3 是余弦相似度
            results = self.db.find_vectors("music_rag", query_vector, limit=count, norm_val=3)
            
            if not results:
                log("No matching songs found.")
                return

            log(f"\nFound {len(results)} matches:")
            for i, res in enumerate(results, 1):
                sim = 1.0 - res.get("dist", 1.0)
                data = res.get("data", {})
                log(f"{i}. {res.get('title')} (Similarity: {sim:.4f})")
                log(f"   Genre: {data.get('genre')} | Emotion: {data.get('emotion')}")
                log(f"   Review: {data.get('review')}")
                log(f"   Path: {data.get('file_path')}")
                log("-" * 20)
        except Exception as e:
            log(f"Search failed: {e}")

if __name__ == "__main__":
    import argparse
    from log import set_log_fn
    set_log_fn(print)
    
    with open(os.path.join(project_root, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    scan_p = subparsers.add_parser("scan")
    scan_p.add_argument("dir", help="Directory to scan")
    
    search_p = subparsers.add_parser("search")
    search_p.add_argument("count", type=int)
    search_p.add_argument("query")

    args = parser.parse_args()
    context = {}
    pool = ThreadPoolExecutor(max_workers=5)
    
    dj = AIDJRag(config, context, pool, source_dir=args.dir if args.command == "scan" else None)
    if args.command == "scan": dj.scan()
    elif args.command == "search": dj.search(args.query, args.count)
    
    pool.shutdown()
