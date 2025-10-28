import json
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import OrderedDict


class RetrievalEngine:
    def __init__(self, db_path, model_name="Qwen/Qwen2-1.5B", cache_size=10):
        self.db_path = db_path
        self.vectors = []
        self.texts = []
        self.model = SentenceTransformer(model_name, device="cpu")
        self._load_embeddings()

        self.cache = OrderedDict()
        self.cache_size = cache_size

    def _load_embeddings(self):
        with open(self.db_path, "r") as f:
            data = json.load(f)
        self.vectors = [np.array(item["embedding"], dtype=np.float32) for item in data]
        self.texts = [item["text"] for item in data]
        print(f"loaded {len(self.vectors)} embeddings from {self.db_path}")

    def _update_cache(self, query, results):
        self.cache[query] = results
        self.cache.move_to_end(query)
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def encode_query(self, query):
        return self.model.encode(query, convert_to_numpy=True)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, top_k=5, threshold=0.7):
        if query in self.cache:
            print("cache hit")
            return self.cache[query]

        query_vector = self.encode_query(query)
        scores = [self.cosine_similarity(v, query_vector) for v in self.vectors]
        ranked = sorted(zip(scores, self.texts), key=lambda x: x[0], reverse=True)
        filtered = [(s, t) for s, t in ranked[:top_k] if s >= threshold]
        self._update_cache(query, filtered)
        return filtered

    def normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-10, None)

    def search_batch(self, queries, top_k=5):
        q_embs = self.model.encode(queries, normalize_embeddings=True)
        sims = np.dot(q_embs, self.vectors.T)
        top_idx = np.argsort(-sims, axis=1)[:, :top_k]
        results = []
        for i, idxs in enumerate(top_idx):
            results.append([(self.texts[j], float(sims[i][j])) for j in idxs])
        return results

    def get_all(self):
        return self.vectors, self.texts
