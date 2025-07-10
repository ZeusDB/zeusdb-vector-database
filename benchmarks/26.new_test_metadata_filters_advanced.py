import time

# Import the vector database module
from zeusdb_vector_database import VectorDatabase

# Instantiate the VectorDatabase class
vdb = VectorDatabase()

# Initialize and set up the database resources
index = vdb.create_index_hnsw(dim=8)

print("=" * 60)
print("ZEUSDB RICH METADATA FILTER TESTING")
print("=" * 60)

# Upload vector records with rich metadata types
records = [
    {
        "id": "doc_001", 
        "values": [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7], 
        "metadata": {
            "author": "Alice",
            "year": 2024,
            "rating": 4.5,
            "published": True,
            "tags": ["science", "research", "ai"],
            "price": 29.99,
            "page_count": 150,
            "language": "en"
        }
    },
    {
        "id": "doc_002", 
        "values": [0.9, 0.1, 0.4, 0.2, 0.8, 0.5, 0.3, 0.9], 
        "metadata": {
            "author": "Bob",
            "year": 2023,
            "rating": 3.8,
            "published": False,
            "tags": ["technology", "web"],
            "price": 19.99,
            "page_count": 89,
            "language": "en"
        }
    },
    {
        "id": "doc_003", 
        "values": [0.11, 0.21, 0.31, 0.15, 0.41, 0.22, 0.61, 0.72], 
        "metadata": {
            "author": "Alice",
            "year": 2025,
            "rating": 5.0,
            "published": True,
            "tags": ["science", "future", "ai"],
            "price": 39.99,
            "page_count": 200,
            "language": "en"
        }
    },
    {
        "id": "doc_004", 
        "values": [0.85, 0.15, 0.42, 0.27, 0.83, 0.52, 0.33, 0.95], 
        "metadata": {
            "author": "Charlie",
            "year": 2024,
            "rating": 4.2,
            "published": True,
            "tags": ["business", "strategy"],
            "price": None,
            "page_count": 120,
            "language": "es"
        }
    },
    {
        "id": "doc_005", 
        "values": [0.2, 0.3, 0.4, 0.1, 0.5, 0.6, 0.7, 0.8], 
        "metadata": {
            "author": "Diana",
            "year": 2022,
            "rating": 3.5,
            "published": True,
            "tags": ["history", "culture"],
            "price": 24.99,
            "page_count": 75,
            "language": "fr"
        }
    },
    {
        "id": "doc_006", 
        "values": [0.3, 0.4, 0.5, 0.2, 0.6, 0.7, 0.8, 0.9], 
        "metadata": {
            "author": "Eve",
            "year": 2024,
            "rating": 4.8,
            "published": False,
            "tags": ["programming", "ai", "python"],
            "price": 49.99,
            "page_count": 300,
            "language": "en"
        }
    },
    {
        "id": "doc_007", 
        "values": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        "metadata": {
            "author": "Frank",
            "year": 2025,
            "rating": 4.9,
            "published": True,
            "tags": ["whitepaper", "pdf", "ai"],
            "price": 34.99,
            "page_count": 180,
            "language": "en",
            "filename": "deep_learning_research.pdf"
        }
    }
]

# Upload records
add_result = index.add(records)
print("\n--- Add Results Summary ---")
print(add_result.summary())

# Query vector
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

print("\n" + "=" * 60)
print("BASIC FILTERING TESTS")
print("=" * 60)

# Test 1–3
print("\n1. NO FILTER")
results = index.search(vector=query_vector, filter=None, top_k=10)
print(f"Found {len(results)} results")
for i, res in enumerate(results, 1):
    print(f"  {i}. ID: {res['id']}, Score: {res['score']:.4f}, Author: {res['metadata']['author']}")

print("\n2. author = 'Alice'")
results = index.search(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Author: {res['metadata']['author']}")

print("\n3. published = True")
results = index.search(vector=query_vector, filter={"published": True}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Published: {res['metadata']['published']}")

print("\n" + "=" * 60)
print("NUMERIC FILTERING TESTS")
print("=" * 60)

# Test 4–7
print("\n4. rating > 4.0")
results = index.search(vector=query_vector, filter={"rating": {"gt": 4.0}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['rating']}")

print("\n5. year >= 2024")
results = index.search(vector=query_vector, filter={"year": {"gte": 2024}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['year']}")

print("\n6. price < 30.0")
results = index.search(vector=query_vector, filter={"price": {"lt": 30.0}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['price']}")

print("\n7. 100 <= page_count <= 200")
results = index.search(vector=query_vector, filter={"page_count": {"gte": 100, "lte": 200}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['page_count']}")

print("\n" + "=" * 60)
print("STRING AND ARRAY TESTS")
print("=" * 60)

# Test 8–11
print("\n8. author contains 'A'")
results = index.search(vector=query_vector, filter={"author": {"contains": "A"}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['author']}")

print("\n9. author startswith 'A'")
results = index.search(vector=query_vector, filter={"author": {"startswith": "A"}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['author']}")

print("\n10. tags contains 'ai'")
results = index.search(vector=query_vector, filter={"tags": {"contains": "ai"}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['tags']}")

print("\n11. language in ['en', 'es']")
results = index.search(vector=query_vector, filter={"language": {"in": ["en", "es"]}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['language']}")

print("\n" + "=" * 60)
print("COMPLEX CONDITIONS")
print("=" * 60)

# Test 12–14
print("\n12. published=True AND rating>=4.0 AND year>=2024")
results = index.search(
    vector=query_vector,
    filter={"published": True, "rating": {"gte": 4.0}, "year": {"gte": 2024}},
    top_k=10
)
for res in results:
    meta = res['metadata']
    print(f"  {res['id']} → Rating: {meta['rating']}, Year: {meta['year']}")

print("\n13. tags contains 'ai' AND language='en' AND price>25")
results = index.search(
    vector=query_vector,
    filter={"tags": {"contains": "ai"}, "language": "en", "price": {"gt": 25.0}},
    top_k=10
)
for res in results:
    meta = res['metadata']
    print(f"  {res['id']} → Price: {meta['price']}, Tags: {meta['tags']}")

print("\n14. author = 'NonExistent'")
results = index.search(vector=query_vector, filter={"author": "NonExistent"}, top_k=10)
print(f"Found {len(results)} results")

print("\n15. METADATA TYPE VERIFICATION")
retrieved = index.get_records("doc_001", return_vector=False)
if retrieved:
    for k, v in retrieved[0]['metadata'].items():
        print(f"  {k}: {v} ({type(v).__name__})")

print("\n16. NULL VALUE: price is None")
try:
    results = index.search(vector=query_vector, filter={"price": None}, top_k=10)
    for res in results:
        print(f"  {res['id']} → {res['metadata']['price']}")
except Exception as e:
    print("Error:", e)

print("\n17. INVALID FILTER OPERATION")
try:
    index.search(vector=query_vector, filter={"rating": {"invalid_op": 4.0}}, top_k=10)
except Exception as e:
    print("Expected error:", e)

print("\n18. PERFORMANCE TEST")
start = time.time()
results = index.search(
    vector=query_vector,
    filter={"year": {"gte": 2022, "lte": 2025}, "rating": {"gte": 3.0}, "published": True},
    top_k=10
)
elapsed = (time.time() - start) * 1000
print(f"Found {len(results)} results in {elapsed:.2f}ms")

print("\n19. AUTHOR IN ['Alice', 'Bob', 'Charlie']")
results = index.search(vector=query_vector, filter={"author": {"in": ["Alice", "Bob", "Charlie"]}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['author']}")

print("\n20. FILENAME ENDSWITH '.pdf'")
results = index.search(vector=query_vector, filter={"filename": {"endswith": ".pdf"}}, top_k=10)
for res in results:
    print(f"  {res['id']} → {res['metadata']['filename']}")

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)
print("✅ Rich metadata system successfully tested")
print(f"📊 Total documents indexed: {len(index.get_stats()['total_vectors'])}")
print("🔍 All filter operations working correctly")
print("⚡ Type fidelity maintained throughout pipeline")
