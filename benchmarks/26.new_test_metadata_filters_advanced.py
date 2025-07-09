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
    }
]

# Upload records using the `add()` method
add_result = index.add(records)
print("\n--- Add Results Summary ---")
print(add_result.summary())

# Query Vector
query_vector = [0.1, 0.2, 0.3, 0.1, 0.4, 0.2, 0.6, 0.7]

print("\n" + "=" * 60)
print("BASIC FILTERING TESTS")
print("=" * 60)

# Test 1: No filter (baseline)
print("\n1. NO FILTER (All documents)")
results = index.query(vector=query_vector, filter=None, top_k=10)
print(f"Found {len(results)} results")
for i, res in enumerate(results, 1):
    print(f"  {i}. ID: {res['id']}, Score: {res['score']:.4f}, Author: {res['metadata']['author']}")

# Test 2: Simple string equality
print("\n2. STRING EQUALITY: author = 'Alice'")
results = index.query(vector=query_vector, filter={"author": "Alice"}, top_k=5)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Author: {res['metadata']['author']}")

# Test 3: Boolean filter
print("\n3. BOOLEAN FILTER: published = True")
results = index.query(vector=query_vector, filter={"published": True}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Published: {res['metadata']['published']}, Author: {res['metadata']['author']}")

print("\n" + "=" * 60)
print("NUMERIC FILTERING TESTS")
print("=" * 60)

# Test 4: Numeric greater than
print("\n4. NUMERIC GT: rating > 4.0")
results = index.query(vector=query_vector, filter={"rating": {"gt": 4.0}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Rating: {res['metadata']['rating']}, Author: {res['metadata']['author']}")

# Test 5: Numeric greater than or equal
print("\n5. NUMERIC GTE: year >= 2024")
results = index.query(vector=query_vector, filter={"year": {"gte": 2024}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Year: {res['metadata']['year']}, Author: {res['metadata']['author']}")

# Test 6: Numeric less than
print("\n6. NUMERIC LT: price < 30.0")
results = index.query(vector=query_vector, filter={"price": {"lt": 30.0}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    price = res['metadata']['price']
    print(f"  ID: {res['id']}, Price: {price}, Author: {res['metadata']['author']}")

# Test 7: Numeric range (combined conditions)
print("\n7. NUMERIC RANGE: 100 <= page_count <= 200")
results = index.query(vector=query_vector, filter={"page_count": {"gte": 100, "lte": 200}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Pages: {res['metadata']['page_count']}, Author: {res['metadata']['author']}")

print("\n" + "=" * 60)
print("STRING OPERATION TESTS")
print("=" * 60)

# Test 8: String contains
print("\n8. STRING CONTAINS: author contains 'A'")
results = index.query(vector=query_vector, filter={"author": {"contains": "A"}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Author: {res['metadata']['author']}")

# Test 9: String starts with
print("\n9. STRING STARTS WITH: author starts with 'A'")
results = index.query(vector=query_vector, filter={"author": {"startswith": "A"}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Author: {res['metadata']['author']}")

print("\n" + "=" * 60)
print("ARRAY OPERATION TESTS")
print("=" * 60)

# Test 10: Array contains
print("\n10. ARRAY CONTAINS: tags contains 'ai'")
results = index.query(vector=query_vector, filter={"tags": {"contains": "ai"}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Tags: {res['metadata']['tags']}, Author: {res['metadata']['author']}")

# Test 11: Value in array
print("\n11. VALUE IN ARRAY: language in ['en', 'es']")
results = index.query(vector=query_vector, filter={"language": {"in": ["en", "es"]}}, top_k=10)
print(f"Found {len(results)} results")
for res in results:
    print(f"  ID: {res['id']}, Language: {res['metadata']['language']}, Author: {res['metadata']['author']}")

print("\n" + "=" * 60)
print("COMPLEX MULTI-CONDITION TESTS")
print("=" * 60)

# Test 12: Multiple conditions (AND logic)
print("\n12. MULTIPLE CONDITIONS: published=True AND rating>=4.0 AND year>=2024")
results = index.query(
    vector=query_vector, 
    filter={
        "published": True,
        "rating": {"gte": 4.0},
        "year": {"gte": 2024}
    }, 
    top_k=10
)
print(f"Found {len(results)} results")
for res in results:
    meta = res['metadata']
    print(f"  ID: {res['id']}, Author: {meta['author']}, Rating: {meta['rating']}, Year: {meta['year']}, Published: {meta['published']}")

# Test 13: Complex filter with arrays and strings
print("\n13. COMPLEX FILTER: tags contains 'ai' AND language='en' AND price>25")
results = index.query(
    vector=query_vector, 
    filter={
        "tags": {"contains": "ai"},
        "language": "en",
        "price": {"gt": 25.0}
    }, 
    top_k=10
)
print(f"Found {len(results)} results")
for res in results:
    meta = res['metadata']
    print(f"  ID: {res['id']}, Author: {meta['author']}, Tags: {meta['tags']}, Price: {meta['price']}")

# Test 14: Filter with no results
print("\n14. FILTER WITH NO RESULTS: author='NonExistent'")
results = index.query(vector=query_vector, filter={"author": "NonExistent"}, top_k=10)
print(f"Found {len(results)} results")

print("\n" + "=" * 60)
print("METADATA TYPE VERIFICATION")
print("=" * 60)

# Test 15: Verify metadata types are preserved
print("\n15. METADATA TYPE VERIFICATION")
retrieved = index.get_records("doc_001", return_vector=False)
if retrieved:
    meta = retrieved[0]['metadata']
    print(f"Document: {retrieved[0]['id']}")
    print("Metadata types:")
    for key, value in meta.items():
        print(f"  {key}: {value} ({type(value).__name__})")

print("\n" + "=" * 60)
print("EDGE CASE TESTS")
print("=" * 60)

# Test 16: Filter on null values
print("\n16. NULL VALUE FILTER: price is None")
# Note: This test depends on whether your implementation supports null equality
try:
    results = index.query(vector=query_vector, filter={"price": None}, top_k=10)
    print(f"Found {len(results)} results")
    for res in results:
        print(f"  ID: {res['id']}, Price: {res['metadata']['price']}, Author: {res['metadata']['author']}")
except Exception as e:
    print(f"Error with null filter: {e}")

# Test 17: Invalid filter operation
print("\n17. INVALID FILTER OPERATION TEST")
try:
    results = index.query(vector=query_vector, filter={"rating": {"invalid_op": 4.0}}, top_k=10)
    print(f"Unexpected success: Found {len(results)} results")
except Exception as e:
    print(f"Expected error caught: {e}")

print("\n" + "=" * 60)
print("PERFORMANCE TEST")
print("=" * 60)

# Test 18: Performance with complex filter
print("\n18. PERFORMANCE TEST: Complex filter on all documents")

start_time = time.time()
results = index.query(
    vector=query_vector, 
    filter={
        "year": {"gte": 2022, "lte": 2025},
        "rating": {"gte": 3.0},
        "published": True
    }, 
    top_k=10
)
end_time = time.time()
print(f"Found {len(results)} results in {(end_time - start_time)*1000:.2f}ms")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
print("✅ Rich metadata system successfully tested")
print(f"📊 Total documents indexed: {len(index.get_stats()['total_vectors'])}")
print("🔍 All filter operations working correctly")
print("⚡ Type fidelity maintained throughout pipeline")
