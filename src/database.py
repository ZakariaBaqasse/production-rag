import os
import asyncpg
from loguru import logger


async def get_db_connection():
    """Establish a connection to the PostgreSQL database."""
    try:
        conn = await asyncpg.connect(
            user=os.environ.get("POSTGRES_USER", "user"),
            password=os.environ.get("POSTGRES_PASSWORD", "password"),
            database=os.environ.get("POSTGRES_DB", "production_rag"),
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            port=os.environ.get("POSTGRES_PORT", 5432),
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


async def init_db():
    """Initialize the database with the required extensions and tables."""
    conn = await get_db_connection()
    try:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create table for storing document pages
        # qwen3-embedding:0.6b produces 1024-dimensional vectors
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_pages (
                id SERIAL PRIMARY KEY,
                page_number INTEGER NOT NULL,
                content TEXT,
                embedding vector(1024),
                metadata JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create an HNSW index for faster similarity search
        # cosine distance is usually good for embeddings (operator <=>)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS document_pages_embedding_idx 
            ON document_pages USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
        """)

        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    finally:
        await conn.close()
