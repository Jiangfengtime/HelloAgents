# 存储包聚合三种职责：SQLite 存文档，Qdrant 存向量索引，Neo4j 存实体关系。
# 连接管理器负责复用实例，不负责自动将三种数据库同步为一个一致事务。

"""存储层模块

按照第8章架构设计的存储层：
- DocumentStore: 文档存储
- QdrantVectorStore: Qdrant向量存储
- Neo4jGraphStore: Neo4j图存储
"""

from .qdrant_store import QdrantVectorStore, QdrantConnectionManager
from .neo4j_store import Neo4jGraphStore
from .document_store import DocumentStore, SQLiteDocumentStore
__all__ = [
    "QdrantVectorStore",
    "QdrantConnectionManager",
    "Neo4jGraphStore",
    "DocumentStore",
    "SQLiteDocumentStore"
]
