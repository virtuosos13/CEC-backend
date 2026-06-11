import json
import os
import numpy as np
import faiss
import google.generativeai as genai
from dotenv import load_dotenv

class CECVectorStore:
    def __init__(self, model_name='models/gemini-embedding-2'):
        self.model_name = model_name
        self.index = None
        self.data = []
        
        # Ensure API key is configured
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

    def build_index(self, processed_json_path):
        """
        Loads processed JSON and builds a FAISS index using Gemini Embeddings.
        """
        if not os.path.exists(processed_json_path):
            print(f"Error: Processed data not found at {processed_json_path}")
            return

        with open(processed_json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        texts = [item['text'] for item in self.data]
        all_embeddings = []
        
        # Batch requests to avoid API limits (100 at a time)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            result = genai.embed_content(
                model=self.model_name,
                content=batch_texts
            )
            all_embeddings.extend(result['embedding'])
            print(f"Processed {min(i+batch_size, len(texts))} / {len(texts)} chunks")
            
        embeddings_array = np.array(all_embeddings).astype('float32')
        
        # Initialize FAISS index
        dimension = embeddings_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        
        print(f"Index built with {len(self.data)} vectors. Dimension: {dimension}")

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
        result = genai.embed_content(
            model=self.model_name,
            content=query
        )
        query_embedding = result['embedding']
        
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        
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
