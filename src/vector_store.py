import os
import chromadb
from chromadb.config import Settings
from src.config import EMBEDDING_MODEL, INDEX_DIR
from sentence_transformers import SentenceTransformer

class SimpleVectorStore:
    def __init__(self):
        # Initialize persistent ChromaDB client with explicit settings & KeyError safety
        chroma_settings = Settings(anonymized_telemetry=False, is_persistent=True)
        try:
            self.client = chromadb.PersistentClient(path=INDEX_DIR, settings=chroma_settings)
        except Exception as e:
            print(f"ChromaDB init warning, clearing system cache: {str(e)}")
            try:
                from chromadb.api.shared_system import SharedSystemClient
                SharedSystemClient._identifier_to_system.clear()
            except Exception:
                pass
            self.client = chromadb.PersistentClient(path=INDEX_DIR, settings=chroma_settings)

        # Create or get collection with cosine similarity space
        self.collection = self.client.get_or_create_collection(
            name="medical_collection",
            metadata={"hnsw:space": "cosine"}
        )
        # Initialize local sentence transformer encoder
        print(f"Loading local embedding model: {EMBEDDING_MODEL}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.chunks = []
        self._sync_chunks_from_db()

    def _sync_chunks_from_db(self):
        """Syncs the internal chunks list with what is stored in ChromaDB."""
        try:
            # Fetch all metadata from collection
            results = self.collection.get()
            self.chunks = []
            if results and "ids" in results:
                for i in range(len(results["ids"])):
                    self.chunks.append({
                        "chunk_id": results["ids"][i],
                        "text": results["documents"][i],
                        "source": results["metadatas"][i].get("source", "unknown"),
                        "page": results["metadatas"][i].get("page", "1"),
                        "theme": results["metadatas"][i].get("theme", "santé générale")
                    })
        except Exception as e:
            print(f"Error syncing vector store chunks: {str(e)}")
            self.chunks = []

    def reset_collection(self):
        """Deletes the collection and recreates it to prevent duplicates on rebuild."""
        try:
            self.client.delete_collection("medical_collection")
        except Exception:
            pass # Collection might not exist yet
            
        self.collection = self.client.create_collection(
            name="medical_collection",
            metadata={"hnsw:space": "cosine"}
        )
        self.chunks = []

    def add_chunks(self, chunks: list) -> bool:
        """
        Generates embeddings locally with Sentence-Transformers and adds them to ChromaDB.
        """
        if not chunks:
            print("No chunks provided to index.")
            return False
            
        # Get list of sources already indexed
        existing_sources = set(chunk["source"] for chunk in self.chunks)
        
        # Filter chunks to only keep those whose source is not already indexed and text is not empty
        new_chunks = [c for c in chunks if c["source"] not in existing_sources and c.get("text", "").strip()]
        
        if not new_chunks:
            print("All documents are already indexed.")
            return True
            
        print(f"Indexing {len(new_chunks)} new chunks (skipping already indexed files)...")
        print(f"Generating embeddings for {len(new_chunks)} chunks using {EMBEDDING_MODEL}...")
        texts = [chunk["text"] for chunk in new_chunks]
        
        # Generate embeddings locally using sentence-transformers
        try:
            embeddings_list = self.encoder.encode(texts, show_progress_bar=True).tolist()
        except Exception as e:
            print(f"Error generating embeddings locally: {str(e)}")
            return False

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in new_chunks]
        documents = [chunk["text"] for chunk in new_chunks]
        
        def sanitize_val(v):
            import math
            if v is None:
                return "inconnu"
            if isinstance(v, float) and math.isnan(v):
                return "inconnu"
            return str(v)
            
        metadatas = [
            {
                "source": sanitize_val(chunk.get("source")),
                "page": sanitize_val(chunk.get("page", "1")),
                "theme": sanitize_val(chunk.get("theme", "santé générale"))
            }
            for chunk in new_chunks
        ]
        
        # Check for any None embeddings and filter them out to prevent SQLite/nan errors
        valid_chunks_data = []
        for i, emb in enumerate(embeddings_list):
            if emb is not None and isinstance(emb, list) and len(emb) > 0:
                valid_chunks_data.append({
                    "id": ids[i],
                    "emb": emb,
                    "doc": documents[i] if documents[i] else "Non disponible",
                    "meta": metadatas[i]
                })
            else:
                print(f"Warning: embedding for chunk index {i} was None or invalid!")
        
        if not valid_chunks_data:
            print("No valid embeddings generated.")
            return False
            
        valid_ids = [item["id"] for item in valid_chunks_data]
        valid_embs = [item["emb"] for item in valid_chunks_data]
        valid_docs = [item["doc"] for item in valid_chunks_data]
        valid_metas = [item["meta"] for item in valid_chunks_data]
        
        # Add to ChromaDB
        self.collection.add(
            ids=valid_ids,
            embeddings=valid_embs,
            metadatas=valid_metas,
            documents=valid_docs
        )
        
        # Sync chunks
        self._sync_chunks_from_db()
        print(f"Successfully added {len(new_chunks)} new chunks to ChromaDB. Total chunks: {len(self.chunks)}")
        return True

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Embeds query locally and queries ChromaDB.
        Returns a list of tuples (chunk, similarity_score).
        """
        if len(self.chunks) == 0:
            print("Vector store is empty.")
            return []
            
        try:
            query_vector = self.encoder.encode([query])[0].tolist()
        except Exception as e:
            print(f"Error embedding query locally: {str(e)}")
            raise e

        # Query ChromaDB
        # ChromaDB returns 'distances'. Since hnsw:space is cosine, distance = 1 - cosine_similarity
        # Therefore, cosine_similarity = 1 - distance
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        search_results = []
        if results and "ids" in results and len(results["ids"][0]) > 0:
            ids = results["ids"][0]
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for i in range(len(ids)):
                chunk = {
                    "chunk_id": ids[i],
                    "text": documents[i],
                    "source": metadatas[i].get("source", "unknown"),
                    "page": metadatas[i].get("page", "1"),
                    "theme": metadatas[i].get("theme", "santé générale")
                }
                # Cosine similarity score
                score = 1.0 - distances[i]
                search_results.append((chunk, score))
                
        return search_results

    def save(self) -> bool:
        # ChromaDB saves automatically as it is configured with PersistentClient
        return True

    def load(self) -> bool:
        # ChromaDB loads automatically upon initialization of PersistentClient
        self._sync_chunks_from_db()
        return len(self.chunks) > 0

    def is_empty(self) -> bool:
        if len(self.chunks) == 0:
            self._sync_chunks_from_db()
        return len(self.chunks) == 0
