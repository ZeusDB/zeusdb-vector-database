#!/usr/bin/env python3
"""
Phase 2: Updated Test for Enhanced Save Method
Tests the actual implementation you have with save() and save_hnsw_graph()
"""

from zeusdb_vector_database import VectorDatabase
import numpy as np
import os

print("🧪 Phase 2: Testing Enhanced Save Method")
print("=" * 50)

# Clean up any existing test directory
test_dir = "test_phase2.zdb"
if os.path.exists(test_dir):
    import shutil
    shutil.rmtree(test_dir)
    print(f"🧹 Cleaned up existing {test_dir}")

print("\n📊 Creating Test Index")
print("-" * 30)

# Create test data
vdb = VectorDatabase()
index = vdb.create("hnsw", dim=128)
print(f"✅ Index created: {index.info()}")

# Add some test vectors
np.random.seed(42)
vectors = np.random.random((50, 128)).astype(np.float32)
test_data = {
    'vectors': vectors.tolist(),
    'ids': [f'test_{i}' for i in range(50)],
    'metadatas': [{'index': i, 'category': 'phase2_test'} for i in range(50)]
}

add_result = index.add(test_data)
print(f"✅ Test data added: {add_result.summary()}")

print("\n💾 Testing Enhanced Save Method")
print("-" * 30)

# This should call your enhanced save method with the HNSW graph dump
print("🔄 Calling index.save() - should see Phase 2 output...")

try:
    index.save(test_dir)
    print("✅ Save method completed without errors!")
except Exception as e:
    print(f"❌ Save method failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n📁 Checking Created Files")
print("-" * 30)

if os.path.exists(test_dir):
    files = os.listdir(test_dir)
    print(f"Files created: {sorted(files)}")
    
    # Check for Phase 1 files (should exist)
    phase1_files = ['config.json', 'manifest.json', 'mappings.bin', 'metadata.json', 'vectors.bin']
    print("\n📋 Phase 1 Files Check:")
    for file in phase1_files:
        if file in files:
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} MISSING")
    
    # Check for Phase 2 files (main test)
    phase2_files = ['hnsw_index.hnsw.graph', 'hnsw_index.hnsw.data']
    print("\n📋 Phase 2 Files Check:")
    for file in phase2_files:
        if file in files:
            size_kb = os.path.getsize(os.path.join(test_dir, file)) / 1024
            if file == 'hnsw_index.hnsw.graph':
                print(f"   🎯 {file} ({size_kb:.1f} KB) - PHASE 2 SUCCESS!")
            else:
                print(f"   ℹ️  {file} ({size_kb:.1f} KB) - (created but ignored)")
        else:
            print(f"   ❌ {file} MISSING")
    
    # Summary
    hnsw_graph_exists = 'hnsw_index.hnsw.graph' in files
    hnsw_data_exists = 'hnsw_index.hnsw.data' in files
    
    print("\n🎯 PHASE 2 RESULTS:")
    if hnsw_graph_exists and hnsw_data_exists:
        print("✅ SUCCESS! Enhanced save method is working!")
        print("   - HNSW graph structure saved (.hnsw.graph)")
        print("   - HNSW data file created (.hnsw.data)")
        print("   - Phase 2 integration successful!")
    elif hnsw_graph_exists:
        print("⚠️  Partial success - graph file created but missing data file")
    else:
        print("❌ FAILED - No HNSW graph files created")
        print("   This means the enhanced save method is not being called")
        print("   or the save_hnsw_graph() method has an issue")

else:
    print(f"❌ Test directory {test_dir} was not created at all!")

print("\n🔍 Expected Output Analysis")
print("-" * 30)

print("If Phase 2 is working, you should see in the output above:")
print("   📊 Saving HNSW graph structure...")
print("   Using [Cosine/L2/L1] distance HNSW")
print("   ✅ HNSW graph saved successfully!")
print("   Files created:")
print("     - hnsw_index.hnsw.graph (graph structure)")
print("     - hnsw_index.hnsw.data (ignored - we use our own data)")
print("   ✅ Phase 2 enhanced save completed successfully!")

print("\nIf you DON'T see this output, then:")
print("   1. The enhanced save method is not in #[pymethods]")
print("   2. VS Code/build issue - need maturin develop --release --force")
print("   3. Method signature or placement issue")

print("\n🚀 Next Steps:")
if os.path.exists(test_dir) and 'hnsw_index.hnsw.graph' in os.listdir(test_dir):
    print("✅ Phase 2 save enhancement working! Ready for Step 6-8!")
else:
    print("❌ Need to fix save method implementation before proceeding")
    print("   Check: method in #[pymethods], rebuild, VS Code issues")

print(f"\n📝 Test directory available for inspection: {test_dir}")