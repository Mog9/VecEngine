import numpy as np
from sklearn.cluster import KMeans


class Indexer:
    def __init__(self, vectors, n_clusters=10):
        self.vectors = np.array(vectors, dtype=np.float32)
        self.n_clusters = n_clusters
        self.kmeans = None
        self.cluster_map = {}

    def build_index(self):
        print(f"building index with {self.n_clusters} clusters")
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=5)
        labels = self.kmeans.fit_predict(self.vectors)
        for i, label in enumerate(labels):
            self.cluster_map.setdefault(label, []).append(i)
        print("index built")

    def search(self, query_vector, top_k=5):
        # for finding nearest cluster center
        cluster_id = np.argmin(
            np.linalg.norm(self.kmeans.cluster_centers_ - query_vector, axis=1)
        )

        # to get vectors in that cluster
        candidate_ids = self.cluster_map.get(cluster_id, [])
        candidates = self.vectors[candidate_ids]

        # compute cosine similarity within the cluster
        sims = np.dot(candidates, query_vector) / (
            np.linalg.norm(candidates, axis=1) * np.linalg.norm(query_vector)
        )

        # get tok_k matches
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(candidate_ids[i], float(sims[i])) for i in top_indices]
