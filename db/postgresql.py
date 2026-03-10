import psycopg
from mod_base import Module as BaseModule
from pydantic import BaseModel,Field
from psycopg_pool import ConnectionPool
from log import log
from typing import Dict,Any

class ModuleConfig(BaseModel):
    db_name : str
    user : str = Field("postgres")
    password : str = Field("")
    host : str = Field("127.0.0.1")
    port : int = Field(5432)
    pool_min_size : int = Field(2)
    pool_max_size : int = Field(10)

class DB(BaseModule):
    namespace = "db_pg"
    connection = None
    config = None
    context : dict
    pool : ConnectionPool

    def __init__(self,external_config, context):
        super().__init__(context)
        self.config = ModuleConfig(**self.cfg(external_config))
        self.context = context

        conn_str = f"dbname={self.config.db_name} "
        conn_str += "" if self.config.user == ""     else f"user={self.config.user} "
        conn_str += "" if self.config.password == "" else f"password={self.config.password} "
        conn_str += "" if self.config.host == ""     else f"host={self.config.host} port={self.config.port}"
        
        log(f"Connecting \"{conn_str}\"")
        self.connection = psycopg.connect(conn_str)
        self.pool = ConnectionPool(conn_str ,min_size=self.config.pool_min_size,max_size=self.config.pool_max_size)
        log("Postgresql interface has been created.")





