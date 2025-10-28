from main import VectorDB

db = VectorDB("vectordb_embed.json")
db.load_from_disk()
db.update_vector("attention.txt_0", {"text": "this is an update"})
db.save_to_disk()

db.delete_vector("attention.txt_0")
db.save_to_disk()
