
from zeusdb_vector_database import VectorDatabase

vdb = VectorDatabase()
index = vdb.create("hnsw", dim=64)

version_num = index.get_version_number()
print(f"Version: {version_num}")

# Should print: Version: 1001