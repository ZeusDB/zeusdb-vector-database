# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create an HNSW index with specific parameters
index = vdb.create(index_type="hnsw", dim=128, space="cosine", m=32, ef_construction=100)

# Outputs the details of the HNSW index
print(index.info())  

# Add index level metadata
index.add_metadata({
  "creator": "Ross Armstrong",
  "version": "0.1",
  "created_at": "2024-01-28T11:35:55Z",
  "index_type": "HNSW",
  "embedding_model": "openai/text-embedding-ada-002",
  "dataset": "docs_corpus_v2",
  "environment": "production",
  "description": "Knowledge base index for customer support articles",
  "num_documents": "15000",
  "tags": "['support', 'docs', '2024']"
})

print(index.get_metadata("creator"))  # "Ross Armstrong"
print(index.get_all_metadata())       # Full dictionary


