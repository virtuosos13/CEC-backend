import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class CECVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.data = []

    def build_index(self, processed_json_path):
        """
        Loads processed JSON and builds a FAISS index.
        """
        if not os.path.exists(processed_json_path):
            print(f"Error: Processed data not found at {processed_json_path}")
            return

        with open(processed_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        texts = [item['text'] for item in self.data]
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))
        
        print(f"Index built with {len(self.data)} vectors.")

    def save_index(self, index_path, data_path):
        faiss.write_index(self.index, index_path)
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f)
        print("Index and data saved.")

    def load_index(self, index_path, data_path):
        self.index = faiss.read_index(index_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print("Index and data loaded.")

    def search(self, query, k=5):
        """
        Searches the index for the top k most relevant chunks.
        """
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.data[idx])
        return results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_json = os.path.join(base_dir, "cec_processed_data.json")
    index_file = os.path.join(base_dir, "cec_index.faiss")
    data_file = os.path.join(base_dir, "cec_chunks.json")
    
    store = CECVectorStore()
    store.build_index(processed_json)
    store.save_index(index_file, data_file)
