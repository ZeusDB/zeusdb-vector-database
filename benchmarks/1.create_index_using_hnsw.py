# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Absolute minimal - uses all defaults
index1 = vdb.create()  # index_type="hnsw", dim=1536, space="cosine", m=16, etc.
print("\n✅ Absolute minimal index creation (all defaults):")
print(index1.info())

# Specify just the index type
# This will use default values for all other parameters
index2 = vdb.create("hnsw")  # dim=1536, space="cosine", m=16, etc.
print("\n✅ Absolute minimal index creation (all defaults):")
print(index1.info())

# Specify just the arguments you need
index3 = vdb.create("hnsw", dim=768, m=32)
print("\n✅ Specify 'dim' and 'm' in index creation:")
print(index2.info())

# Specify all arguments explicitly
index4 = vdb.create(
    "hnsw",
    dim=128,
    space="cosine",
    m=32,
    ef_construction=100,
    expected_size=5
)
print("\n✅ Specify all arguments explicitly:")
print(index4.info())
