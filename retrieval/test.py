from main import RetrievalEngine
from indexing import Indexer

engine = RetrievalEngine("vectordb_embed.json", cache_size=10)

query = "how does a transformer understand context?"
_ = engine.search(query)  # first call (cache miss)
_ = engine.search(query)  # second call (cache hit)
