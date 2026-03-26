import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from .config import CHROMA_PATH, EMBED_MODEL_NAME
from .preprocessing import row_to_string

class VectorStore:
    def __init__(self):
        """Initializes the Embedding Model and Persistent ChromaDB Client."""
        print("🛠️ Initializing Vector Store...")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        
        # Ensure the directory exists
        if not os.path.exists(CHROMA_PATH):
            os.makedirs(CHROMA_PATH)
            
        # PersistentClient ensures data is saved to your disk
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        
        # Get or create the collection
        self.collection = self.client.get_or_create_collection(name="nsl_kdd_full")

    def add_to_index(self, df, batch_size=2000):
        """
        Indexes the entire dataframe in batches.
        The 'Skipping' check has been removed to force a full re-index.
        """
        total_rows = len(df)
        
        print(f"🚀 Starting Full Indexing of {total_rows} rows...")
        print(f"📦 Batch Size: {batch_size} | Model: {EMBED_MODEL_NAME}")

        # Using tqdm for a visual progress bar in your terminal
        for i in tqdm(range(0, total_rows, batch_size), desc="Indexing Progress"):
            batch = df.iloc[i : i + batch_size]
            
            # 1. Convert rows to strings (Semantic representation)
            documents = batch.apply(row_to_string, axis=1).tolist()
            
            # 2. Extract labels for metadata (Helps the LLM verify the match)
            metadatas = [{"label": str(row['label'])} for _, row in batch.iterrows()]
            
            # 3. Create unique IDs for every single row
            ids = [f"id_{j}" for j in range(i, i + len(batch))]

            # 4. Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
        
        print(f"✅ Successfully indexed {self.collection.count()} total records.")

    def query_similar_cases(self, query_row, n_results=5):
        """
        Searches the database for the most similar historical network connections.
        """
        query_text = row_to_string(query_row)
        
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results