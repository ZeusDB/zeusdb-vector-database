#!/usr/bin/env python3
"""
Step-by-Step Debug Test
Tests each phase individually to isolate any issues
"""

from zeusdb_vector_database import VectorDatabase
# import numpy as np
import os
# import shutil

print("🔧 Step-by-Step Debug Test")
print("=" * 40)

test_dir = "debug_test.zdb"

# Clean up
# if os.path.exists(test_dir):
#     shutil.rmtree(test_dir)

print("Step 1: Create minimal test index")
print("-" * 30)

try:
    vdb = VectorDatabase()
    index = vdb.create("hnsw", dim=4, space="cosine")  # Very simple: 4D vectors
    print("✅ Index created")
    
    # Add just 3 vectors
    test_data = {
        'vectors': [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        'ids': ['vec_a', 'vec_b', 'vec_c'],
        'metadatas': [{'type': 'test'}, {'type': 'test'}, {'type': 'test'}]
    }
    
    result = index.add(test_data)
    print(f"✅ Added vectors: {result.summary()}")
    
except Exception as e:
    print(f"❌ Step 1 failed: {e}")
    exit(1)

print("\nStep 2: Test save functionality")
print("-" * 30)

try:
    index.save(test_dir)
    print("✅ Save completed")
    
    # Check files
    files = os.listdir(test_dir)
    print(f"Files created: {sorted(files)}")
    
except Exception as e:
    print(f"❌ Step 2 failed: {e}")
    exit(1)

print("\nStep 3: Test load functionality")
print("-" * 30)

try:
    print("🔄 Attempting load...")
    loaded_index = vdb.load(test_dir)
    print("🎉 SUCCESS! Load completed!")
    
    # Quick verification
    stats = loaded_index.get_stats()
    print(f"Loaded index has {stats['total_vectors']} vectors")
    
    # Test search
    results = loaded_index.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    print(f"Search returned {len(results)} results")
    if results:
        print(f"Top result: {results[0]['id']} (score: {results[0]['score']:.6f})")
    
    print("✅ All functionality verified!")
    
except Exception as e:
    print(f"❌ Step 3 failed: {e}")
    
    # Detailed error analysis
    error_str = str(e)
    print("\n🔍 Error Analysis:")
    print(f"Error message: {error_str}")
    
    if "new_empty" in error_str:
        print("💡 Issue: new_empty() method not found")
        print("   Solution: Add the method to hnsw_index.rs outside #[pymethods]")
        
    elif "set_vectors" in error_str or "set_" in error_str:
        print("💡 Issue: Setter methods not found") 
        print("   Solution: Add setter methods to hnsw_index.rs")
        
    elif "reconstruct_index_simple" in error_str:
        print("💡 Issue: Reconstruction function not found")
        print("   Solution: Add reconstruction functions to persistence.rs")
        
    elif "Failed to load" in error_str:
        print("💡 Issue: Component loading problem")
        print("   Check: Helper functions in persistence.rs")
        
    else:
        print("💡 Issue: Other implementation problem")
        print("   Check: All artifacts integrated correctly")
    
    print("\n📋 Integration Checklist:")
    print("□ Added new_empty() to hnsw_index.rs")
    print("□ Added setter methods to hnsw_index.rs") 
    print("□ Added reconstruction functions to persistence.rs")
    print("□ Rebuilt with: maturin develop --release --force")
    print("□ Restarted Python interpreter")

print(f"\n📝 Debug test data in: {test_dir}")