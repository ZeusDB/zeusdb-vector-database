#import pytest
import numpy as np
from zeusdb_vector_database import VectorDatabase

def test_numpy_parsing_debug():
    """Isolated test to debug NumPy parsing specifically"""
    print("\n" + "="*60)
    print("🧪 NUMPY PARSING DEBUG TEST")
    print("="*60)
    
    # Create index
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, expected_size=10)
    print("✅ Created index with dim=4")
    
    # Create simple test data
    test_data = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2]
    ], dtype=np.float32)
    
    print(f"🔍 Test data shape: {test_data.shape}")
    print(f"🔍 Test data dtype: {test_data.dtype}")
    print(f"🔍 Test data:\n{test_data}")
    
    # Test Format 5: NumPy array directly
    print("\n📦 Testing Format 5 (NumPy array)...")
    add_result = index.add({
        "ids": ["numpy_doc1", "numpy_doc2", "numpy_doc3"],
        "embeddings": test_data,
        "metadatas": [{"source": "test"}, {"source": "test"}, {"source": "test"}]
    })
    
    print("📊 Add result:")
    print(f"   - Inserted: {add_result.total_inserted}")
    print(f"   - Errors: {add_result.total_errors}")
    print(f"   - Error messages: {add_result.errors}")
    print(f"   - Vector shape: {add_result.vector_shape}")
    print(f"   - Success: {add_result.is_success()}")
    
    # Check what's actually in the index
    print("\n🔍 Checking index contents...")
    stats = index.get_stats()
    print(f"   - Total vectors in index: {stats['total_vectors']}")
    
    # List all records
    all_records = index.list(number=20)
    print(f"   - Records found: {len(all_records)}")
    for i, (record_id, metadata) in enumerate(all_records):
        print(f"     {i}: ID='{record_id}', metadata={metadata}")
    
    # Try to get specific records
    print("\n🔍 Trying to get specific records...")
    try:
        specific_records = index.get_records(["numpy_doc1", "numpy_doc2", "numpy_doc3"])
        print(f"   - Found {len(specific_records)} specific records")
        for record in specific_records:
            print(f"     - ID: {record['id']}")
            print(f"     - Vector length: {len(record.get('vector', []))}")
            print(f"     - Metadata: {record['metadata']}")
    except Exception as e:
        print(f"   - Error getting specific records: {e}")
    
    # Test search to see if vectors are actually indexed
    print("\n🔍 Testing search functionality...")
    try:
        query = [0.1, 0.2, 0.3, 0.4]
        search_results = index.search(query, top_k=5)
        print(f"   - Search found {len(search_results)} results")
        for i, result in enumerate(search_results):
            print(f"     {i}: ID='{result['id']}', score={result['score']:.4f}")
    except Exception as e:
        print(f"   - Search error: {e}")
    
    print("\n" + "="*60)
    print("🧪 DEBUG TEST COMPLETE")
    print("="*60)

def test_numpy_step_by_step():
    """Even more granular test"""
    print("\n🔬 STEP-BY-STEP NUMPY TEST")
    
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=2, expected_size=5)
    
    # Step 1: Test single NumPy vector
    print("\n📋 Step 1: Single NumPy vector")
    single_vector = np.array([0.1, 0.2], dtype=np.float32)
    print(f"Single vector: {single_vector}, shape: {single_vector.shape}")
    
    # Step 2: Test 2D NumPy array with 1 row
    print("\n📋 Step 2: 2D NumPy array (1 row)")
    single_2d = np.array([[0.3, 0.4]], dtype=np.float32)
    print(f"Single 2D: {single_2d}, shape: {single_2d.shape}")
    
    try:
        result1 = index.add({
            "ids": ["single_2d"],
            "embeddings": single_2d,
            "metadatas": [{"test": "single_2d"}]
        })
        print(f"✅ Single 2D result: inserted={result1.total_inserted}, errors={result1.total_errors}")
    except Exception as e:
        print(f"❌ Single 2D failed: {e}")
    
    # Step 3: Test 2D NumPy array with 2 rows
    print("\n📋 Step 3: 2D NumPy array (2 rows)")
    double_2d = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
    print(f"Double 2D: {double_2d}, shape: {double_2d.shape}")
    
    try:
        result2 = index.add({
            "ids": ["double_1", "double_2"],
            "embeddings": double_2d,
            "metadatas": [{"test": "double_1"}, {"test": "double_2"}]
        })
        print(f"✅ Double 2D result: inserted={result2.total_inserted}, errors={result2.total_errors}")
    except Exception as e:
        print(f"❌ Double 2D failed: {e}")
    
    # Check final state
    print(f"\n📊 Final index state: {index.get_stats()['total_vectors']} vectors")

if __name__ == "__main__":
    test_numpy_parsing_debug()
    test_numpy_step_by_step()