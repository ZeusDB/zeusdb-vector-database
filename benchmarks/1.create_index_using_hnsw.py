from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create_index_hnsw(dim = 1536, space = "cosine", M = 16, ef_construction = 200, expected_size=5)

 # Outputs the details of the HNSW index
print(index.info()) 
 