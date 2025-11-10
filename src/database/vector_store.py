"""Vector database for storing and retrieving queries based on semantic similarity."""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger


class VectorStore:
    """
    Vector database for storing query embeddings and retrieving similar queries.
    Uses ChromaDB for persistent storage and sentence transformers for embeddings.
    """

    def __init__(
        self,
        db_path: str = "./data/vector_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        collection_name: str = "queries",
    ):
        """
        Initialize the vector store.

        Args:
            db_path: Path to store the ChromaDB database
            embedding_model: Name of the sentence transformer model to use
            collection_name: Name of the collection to store queries
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"VectorStore initialized with {self.collection.count()} existing queries")

    def add_query(
        self,
        query_id: str,
        query_text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a query to the vector database.

        Args:
            query_id: Unique identifier for the query
            query_text: The query text
            metadata: Optional metadata to store with the query
        """
        # Generate embedding
        embedding = self.embedding_model.encode(query_text).tolist()

        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "query_text": query_text,
            "timestamp": datetime.now().isoformat(),
        })

        # Add to collection
        self.collection.add(
            ids=[query_id],
            embeddings=[embedding],
            metadatas=[meta],
            documents=[query_text],
        )

        logger.debug(f"Added query {query_id} to vector store")

    def find_similar_queries(
        self,
        query_text: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Find queries similar to the input query.

        Args:
            query_text: The query to find similar queries for
            top_k: Number of similar queries to return
            min_similarity: Minimum similarity score (0-1)

        Returns:
            List of dictionaries containing similar queries with metadata and scores
        """
        # Generate embedding for query
        embedding = self.embedding_model.encode(query_text).tolist()

        # Query the collection
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

        # Format results
        similar_queries = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                # ChromaDB returns distance, convert to similarity (1 - distance for cosine)
                distance = results["distances"][0][i]
                similarity = 1 - distance

                if similarity >= min_similarity:
                    similar_queries.append({
                        "query_id": results["ids"][0][i],
                        "query_text": results["documents"][0][i],
                        "similarity": similarity,
                        "metadata": results["metadatas"][0][i],
                    })

        return similar_queries

    def update_query_metadata(
        self,
        query_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Update metadata for an existing query.

        Args:
            query_id: The query ID to update
            metadata: New metadata to merge with existing
        """
        # Get existing metadata
        existing = self.collection.get(ids=[query_id])

        if not existing["ids"]:
            logger.warning(f"Query {query_id} not found in vector store")
            return

        # Merge metadata
        current_meta = existing["metadatas"][0]
        current_meta.update(metadata)

        # Update
        self.collection.update(
            ids=[query_id],
            metadatas=[current_meta],
        )

        logger.debug(f"Updated metadata for query {query_id}")

    def get_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a query by ID.

        Args:
            query_id: The query ID

        Returns:
            Dictionary with query data or None if not found
        """
        results = self.collection.get(ids=[query_id])

        if not results["ids"]:
            return None

        return {
            "query_id": results["ids"][0],
            "query_text": results["documents"][0],
            "metadata": results["metadatas"][0],
        }

    def delete_query(self, query_id: str) -> None:
        """Delete a query from the vector store."""
        self.collection.delete(ids=[query_id])
        logger.debug(f"Deleted query {query_id}")

    def count(self) -> int:
        """Get the total number of queries stored."""
        return self.collection.count()

    def clear(self) -> None:
        """Clear all queries from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Cleared all queries from vector store")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        return {
            "total_queries": self.count(),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "db_path": self.db_path,
        }
