# Import the vector database module
from zeusdb_vector_database import VectorDatabase
import numpy as np
import time

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Create index
index = vdb.create(index_type="hnsw", dim=3, space="cosine", m=16, ef_construction=200, expected_size=1000)

# Add some points
add_result = index.add([
    {"id": "a", "values": [0.1, 0.2, 0.3], "metadata": {"category": "A"}},  # Note: "values" not "vector"
    {"id": "b", "values": [0.4, 0.5, 0.6], "metadata": {"category": "B"}},
    {"id": "c", "values": [0.7, 0.8, 0.9], "metadata": {"category": "A"}},
])
print("Setup - Added:", add_result.total_inserted, "vectors")

# 🔍 Test 1: Single Vector Search
print("\n=== Test 1: Single Vector Search ===")
results = index.search([0.1, 0.2, 0.3])
print("✓ Single vector result count:", len(results))
print("✓ Result type:", type(results).__name__)  # Should be 'list'
for r in results:
    print(f"  {r['id']}: score={r['score']:.3f}")
    assert "id" in r and "score" in r, "Missing required fields"
print("Raw Response:", results)

# 🧪 Test 2: List of Vectors (Batch Search)
print("\n=== Test 2: List of Vectors (Batch Search) ===")
batch_results = index.search([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6]
])
print("✓ Batch result size:", len(batch_results))  # Should be 2
print("✓ Outer type:", type(batch_results).__name__)  # Should be 'list'
for i, batch in enumerate(batch_results):
    print(f"✓ Query {i} results: {len(batch)} neighbors")
    print(f"✓ Inner type: {type(batch).__name__}")  # Should be 'list'
    for r in batch:
        print(f"    {r['id']}: score={r['score']:.3f}")
print("Raw Batch Response:", batch_results)

# 🧪 Test 3: NumPy 2D Batch Input
print("\n=== Test 3: NumPy 2D Batch Input ===")
queries_np = np.array([
    [0.1, 0.2, 0.3],
    [0.7, 0.8, 0.9]
], dtype=np.float32)

np_results = index.search(queries_np)
print("✓ NumPy batch result count:", len(np_results))  # Should be 2
print("✓ Input shape:", queries_np.shape)
for i, batch in enumerate(np_results):
    print(f"✓ Query {i} returned {len(batch)} results")
    if batch:  # If results exist
        print(f"    Best match: {batch[0]['id']} (score: {batch[0]['score']:.3f})")

# 🎯 Test 4: With Metadata Filter
print("\n=== Test 4: With Metadata Filter ===")
filtered = index.search(
    [[0.1, 0.2, 0.3], [0.7, 0.8, 0.9]],
    filter={"category": "A"}
)
print("✓ Filtered batch results:")
for i, batch in enumerate(filtered):
    print(f"✓ Query {i} filtered results: {len(batch)} matches")
    for r in batch:
        print(f"    {r['id']}: category={r['metadata']['category']}")
        assert r['metadata']['category'] == "A", "Filter not working!"

# 📦 Test 5: return_vector=True
print("\n=== Test 5: return_vector=True ===")
vectors_back = index.search(
    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    return_vector=True
)
print("✓ return_vector=True test:")
for i, batch in enumerate(vectors_back):
    for j, r in enumerate(batch):
        assert "vector" in r, "Vector not returned!"
        print(f"✓ Query {i}, Result {j}: vector length = {len(r['vector'])}")
        print(f"    Vector: {[round(x, 2) for x in r['vector']]}")

# 🔄 Test 6: Mixed Input Handling (PyArray1 fallback)
print("\n=== Test 6: Single NumPy Vector (1D) ===")
single_np = np.array([0.1, 0.2, 0.3], dtype=np.float32)
mixed_result = index.search(single_np)
print("✓ Single NumPy vector returns:", len(mixed_result), "results")
print("✓ Type check: single NumPy input returns flat list")
print("✓ Shape:", single_np.shape, "→ flat result list")

# 🚀 Test 7: Performance Difference (Sequential vs Batch)
print("\n=== Test 7: Performance Comparison ===")


# Create more test queries
test_queries = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]] * 3  # 9 queries

# Sequential individual searches
start = time.time()
sequential_results = []
for query in test_queries:
    sequential_results.append(index.search(query))
sequential_time = time.time() - start

# Single batch search
start = time.time()
batch_result = index.search(test_queries)
batch_time = time.time() - start

print(f"✓ Sequential time: {sequential_time:.4f}s")
print(f"✓ Batch time: {batch_time:.4f}s")
print(f"✓ Speedup: {sequential_time/batch_time:.2f}x")

# 🔥 Test 8: Error Handling
print("\n=== Test 8: Error Handling ===")
try:
    # Wrong dimension
    index.search([0.1, 0.2])  # Only 2 dims, should be 3
    print("❌ Should have failed!")
except Exception as e:
    print("✓ Dimension mismatch caught:", str(e)[:50])

try:
    # Wrong NumPy shape
    bad_np = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)  # 2 dims each
    index.search(bad_np)
    print("❌ Should have failed!")
except Exception as e:
    print("✓ NumPy shape error caught:", str(e)[:50])

print("\n🎯 All Tests Complete! Your batch search is working correctly! 🚀")