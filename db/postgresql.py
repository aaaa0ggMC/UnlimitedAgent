import psycopg
from psycopg import sql
import json
from mod_base import Module as BaseModule
from pydantic import BaseModel, Field
from psycopg_pool import ConnectionPool
from log import log
from typing import Any, List, Optional, Dict
from pgvector import Vector

class ModuleConfig(BaseModel):
    db_name : str
    user : str = Field("postgres")
    password : str = Field("")
    host : str = Field("127.0.0.1")
    port : int = Field(5432)
    pool_min_size : int = Field(2)
    pool_max_size : int = Field(10)

class DB(BaseModule):
    shared_namespace = "_shared_db_pg"
    namespace = "db_pg"
    config = None
    context : dict
    pool : ConnectionPool

    def __init__(self, external_config, context,namespace = "db_pg"):
        self.namespace = namespace
        super().__init__(context)
        self.config = ModuleConfig(**self.cfg(external_config))
        self.context = context

        conn_str = f"dbname={self.config.db_name} "
        conn_str += "" if self.config.user == ""     else f"user={self.config.user} "
        conn_str += "" if self.config.password == "" else f"password={self.config.password} "
        conn_str += "" if self.config.host == ""     else f"host={self.config.host} port={self.config.port}"
        
        log(f"Connecting \"{conn_str}\"")
        self.pool = ConnectionPool(conn_str, min_size=self.config.pool_min_size, max_size=self.config.pool_max_size)
        log("Postgresql interface has been created.")

    def store(self, form: str, value: Any, pk: Any = None, try_vectorize: bool = False):
        """
        动态创建表单并存储数据。支持 Upsert 和 Vector 的高阶 JSONB 结构。
        """
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif hasattr(value, "dict"):
            value = value.dict()
        elif not isinstance(value, dict):
            raise ValueError("store 方法的 value 必须是 dict 或 Pydantic 模型")

        col_defs = {}  
        insert_cols = []
        insert_vals: List[Any] = []
        vector_col_name = None 
        
        if try_vectorize and "vector" in value and "data" in value:
            vec_val = value["vector"]
            if hasattr(vec_val, "to_list"): 
                vec_val = vec_val.to_list()
            dim = len(vec_val)
            
            col_defs["vector"] = f"VECTOR({dim})"
            insert_cols.append("vector")
            insert_vals.append(str(vec_val))
            vector_col_name = "vector"
            
            col_defs["data"] = "JSONB"
            insert_cols.append("data")
            insert_vals.append(json.dumps(value["data"], ensure_ascii=False))
            
            for k, v in value.items():
                if k not in ("vector", "data"):
                    pg_type, v_parsed = self._infer_type(v, try_vectorize)
                    col_defs[k] = pg_type
                    insert_cols.append(k)
                    insert_vals.append(v_parsed)
        else:
            for k, v in value.items():
                pg_type, v_parsed = self._infer_type(v, try_vectorize)
                col_defs[k] = pg_type
                insert_cols.append(k)
                insert_vals.append(v_parsed)
                if "VECTOR" in pg_type:
                    vector_col_name = k

        if pk is not None:
            if pk in col_defs:
                col_defs[pk] += " PRIMARY KEY"
            else:
                col_defs[pk] = "SERIAL PRIMARY KEY"

        cols_sql = [
            sql.SQL("{} {}").format(sql.Identifier(k), sql.SQL(t)) 
            for k, t in col_defs.items()
        ]
        create_sql = sql.SQL("CREATE TABLE IF NOT EXISTS {} ({});").format(
            sql.Identifier(form),
            sql.SQL(", ").join(cols_sql)
        )
        
        insert_cols_sql = [sql.Identifier(c) for c in insert_cols]
        placeholders_sql = [sql.Placeholder()] * len(insert_cols)
        
        insert_base = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(form),
            sql.SQL(", ").join(insert_cols_sql),
            sql.SQL(", ").join(placeholders_sql)
        )
        
        # 修复 Pylance 报错：使用 sql.Composed 规范拼接
        if pk is not None and pk in insert_cols:
            update_clauses = [
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
                for c in insert_cols if c != pk
            ]
            if update_clauses:
                upsert_sql = sql.SQL(" ON CONFLICT ({}) DO UPDATE SET {}").format(
                    sql.Identifier(pk),
                    sql.SQL(", ").join(update_clauses)
                )
            else:
                upsert_sql = sql.SQL(" ON CONFLICT ({}) DO NOTHING").format(sql.Identifier(pk))
            
            final_insert_query = sql.Composed([insert_base, upsert_sql])
        else:
            final_insert_query = insert_base

        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                if try_vectorize or vector_col_name:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                cur.execute(create_sql)
                
                # 自动创建 GIN 索引加速 JSONB 查询
                if "data" in col_defs and col_defs["data"] == "JSONB":
                    idx_name_gin = f"idx_{form}_data_gin"
                    create_gin_sql = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (data);").format(
                        sql.Identifier(idx_name_gin),
                        sql.Identifier(form)
                    )
                    cur.execute(create_gin_sql)

                # 自动创建 HNSW 向量索引
                if vector_col_name:
                    idx_name_hnsw = f"idx_{form}_{vector_col_name}_hnsw"
                    create_idx_sql = sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw ({} vector_l2_ops);"
                    ).format(
                        sql.Identifier(idx_name_hnsw),
                        sql.Identifier(form),
                        sql.Identifier(vector_col_name)
                    )
                    cur.execute(create_idx_sql)

                cur.execute(final_insert_query, insert_vals)
            conn.commit()

    def _infer_type(self, v: Any, try_vectorize: bool):
        if isinstance(v, int): 
            if v > 2147483647:
                return "BIGINT", v
            return "INTEGER", v
        if isinstance(v, float): return "DOUBLE PRECISION", v
        if isinstance(v, bool): return "BOOLEAN", v
        if isinstance(v, Vector):
            return f"VECTOR({len(v.to_list())})", str(v.to_list())
        if try_vectorize and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            return f"VECTOR({len(v)})", str(v)
        return "TEXT", str(v)

    def get(self, form: str, limit: int = 0, pks: Optional[List[Any]] = None):
        """提取表单数据。"""
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);", (form,))
                exists_row = cur.fetchone()
                if not exists_row or not exists_row[0]:
                    log(f"Table '{form}' does not exist.")
                    return []

                # 修复 Pylance 报错：使用 sql.Composed
                query_components = [sql.SQL("SELECT * FROM {}").format(sql.Identifier(form))]
                params: List[Any] = []

                if pks is not None and len(pks) > 0:
                    cur.execute("""
                        SELECT a.attname
                        FROM pg_index i
                        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                        WHERE i.indrelid = %s::regclass AND i.indisprimary;
                    """, (form,))
                    pk_row = cur.fetchone()
                    pk_col = pk_row[0] if pk_row else "id"
                    
                    query_components.append(sql.SQL(" WHERE {} = ANY(%s)").format(sql.Identifier(pk_col)))
                    params.append(pks)
                
                if limit > 0:
                    query_components.append(sql.SQL(" LIMIT %s"))
                    params.append(limit)
                    
                final_query = sql.Composed(query_components)
                cur.execute(final_query, params)
                
                columns = [] if not cur.description else [ desc.name for desc in cur.description ]
                return [dict(zip(columns, row)) for row in cur.fetchall()]

    def find_vectors(self, form: str, vector: Vector, norm_val: int = 2, limit: int = 10, threshold: Optional[float] = None, meta_filter: Optional[Dict] = None):
        """
        norm_val: 1 = 内积 (<+>), 2 = L2距离 (<->), 3 = 余弦相似度 (<=>)
        """
        # 默认改回 2 (L2)，防止测试零向量时抛出 NaN
        op_map = {1: sql.SQL("<+>"), 2: sql.SQL("<->"), 3: sql.SQL("<=>")}
        op_sql = op_map.get(norm_val, sql.SQL("<->"))
        
        with self.pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s);", (form,))
                exists_res = cur.fetchone()
                if not exists_res or not exists_res[0]: return []

                cur.execute("""
                    SELECT a.attname
                    FROM pg_attribute a
                    JOIN pg_type t ON a.atttypid = t.oid
                    WHERE a.attrelid = %s::regclass 
                      AND t.typname = 'vector'
                      AND a.attnum > 0 
                      AND NOT a.attisdropped
                    LIMIT 1;
                """, (form,))
                col_row = cur.fetchone()
                if not col_row:
                    raise ValueError(f"No vector column found in table '{form}'")
                vec_col = col_row[0]
                
                vec_str = str(vector.to_list()) if hasattr(vector, 'to_list') else str(vector)

                # 修复 Pylance 报错：显式声明 List[Any]
                params: List[Any] = [vec_str]
                
                base_query = sql.SQL("SELECT *, {col} {op} %s::vector AS dist FROM {table}").format(
                    col=sql.Identifier(vec_col),
                    op=op_sql,
                    table=sql.Identifier(form)
                )
                
                where_clauses = []

                if threshold is not None:
                    where_clauses.append(sql.SQL("({col} {op} %s::vector) < %s").format(
                        col=sql.Identifier(vec_col),
                        op=op_sql
                    ))
                    params.extend([vec_str, threshold])

                if meta_filter is not None:
                    where_clauses.append(sql.SQL("data @> %s::jsonb"))
                    params.append(json.dumps(meta_filter, ensure_ascii=False))

                # 修复 Pylance 报错：规范拼接
                query_components = [base_query]
                
                if where_clauses:
                    query_components.append(sql.SQL(" WHERE "))
                    query_components.append(sql.SQL(" AND ").join(where_clauses))

                query_components.append(sql.SQL(" ORDER BY dist ASC LIMIT %s"))
                params.append(limit)

                final_query = sql.Composed(query_components)

                cur.execute(final_query, params)
                columns = [desc.name for desc in cur.description] if cur.description else []
                return [dict(zip(columns, row)) for row in cur.fetchall()]