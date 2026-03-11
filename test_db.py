from db import postgresql
import json
from log import log
from pgvector import Vector

def main():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            external_config = json.load(f)
    except FileNotFoundError:
        log("错误: 找不到 config.json 文件")
        return
    context = {}
    db = postgresql.DB(external_config,context)

    db.store("vectors",{
        "vector" : Vector([2,2,3,4,5]),
        "data" : "真的吗?2.0"
    },try_vectorize=True)

    print(db.find_vectors("vectors",Vector([0.01,0,0,0,0]),limit=2,norm_val = 1))

if __name__ == "__main__":
    main()