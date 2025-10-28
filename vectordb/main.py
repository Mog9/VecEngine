import json, os


class VectorDB:
    def __init__(self, path="vector_db.json"):
        self.path = path
        self.vectors = []

    def load_from_disk(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.vectors = json.load(f)
            print(f"loaded {len(self.vectors)} vectors from {self.path}")
        else:
            print("no existing database found")

    def save_to_disk(self):
        with open(self.path, "w") as f:
            json.dump(self.vectors, f, indent=2)
        print(f"saved {len(self.vectors)} vectors to {self.path}")

    def add_from_file(self, file_path):
        with open(file_path, "r") as f:
            new_vectors = json.load(f)
        added = 0
        for item in new_vectors:
            vid = item.get("id") or f"{item['source']}_{item['chunk_id']}"
            if not any(
                (v.get("id") or f"{v['source']}_{v['chunk_id']}") == vid
                for v in self.vectors
            ):
                item["id"] = vid
                self.vectors.append(item)
                added += 1
        print(f"added {added} new vectors from {file_path}")

    def update_vector(self, vid, new_data):
        for v in self.vectors:
            if v.get("id") == vid:
                v.update(new_data)
                print(f"updated vector {vid}")
                return
        print(f"vector {vid} not found")

    def delete_vector(self, vid):
        before = len(self.vectors)
        self.vectors = [v for v in self.vectors if v.get("id") != vid]
        if len(self.vectors) < before:
            print(f"deleted vector {vid}")
        else:
            print(f"vector {vid} not found")

    def __len__(self):
        return len(self.vectors)
