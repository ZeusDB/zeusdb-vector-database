#!/usr/bin/env python3
"""
Phase 1 Persistence Test
Tests the save functionality for ZeusDB vector database persistence.

"""

import numpy as np
import os
#import shutil
import json
from pathlib import Path

# Import your vector database
from zeusdb_vector_database import VectorDatabase

print("🚀 Starting Phase 1 Persistence Tests")
print("=" * 50)

# # Clean up any existing test directories - I turned this off to be safe. Will manually clean up after each tests
test_dirs = ["test_raw.zdb", "test_quantized.zdb", "test_quantized_only.zdb"]
# for test_dir in test_dirs:
#     if os.path.exists(test_dir):
#         shutil.rmtree(test_dir)
#         print(f"🧹 Cleaned up existing {test_dir}")

print("\n📊 Test Data Generation")
print("-" * 30)

# Generate test data
dim = 384  # Smaller dimension for faster testing
num_vectors = 1500  # Enough to trigger quantization training (if enabled)

# Create diverse test vectors
np.random.seed(42)  # For reproducible results
vectors = []
ids = []
metadatas = []

for i in range(num_vectors):
    # Create diverse vector patterns
    if i < 500:
        # Cluster 1: Around [1, 1, 1, ...]
        vector = np.random.normal(1.0, 0.3, dim).astype(np.float32)
    elif i < 1000:
        # Cluster 2: Around [-1, -1, -1, ...]
        vector = np.random.normal(-1.0, 0.3, dim).astype(np.float32)
    else:
        # Cluster 3: Around [0, 0, 0, ...]
        vector = np.random.normal(0.0, 0.5, dim).astype(np.float32)
    
    vectors.append(vector.tolist())
    ids.append(f"doc_{i:04d}")
    metadatas.append({
        "cluster": "A" if i < 500 else "B" if i < 1000 else "C",
        "index": i,
        "category": "test_data",
        "value": float(i * 0.1)
    })

print(f"✅ Generated {len(vectors)} test vectors")
print(f"   - Dimension: {dim}")
print(f"   - Vector range: {np.min(vectors):.3f} to {np.max(vectors):.3f}")
print(f"   - Clusters: A({sum(1 for m in metadatas if m['cluster'] == 'A')}), "
      f"B({sum(1 for m in metadatas if m['cluster'] == 'B')}), "
      f"C({sum(1 for m in metadatas if m['cluster'] == 'C')})")

# Test data structure
batch_data = {
    "vectors": vectors,
    "ids": ids,
    "metadatas": metadatas
}

print("\n🧪 TEST 1: Raw Vector Index (No Quantization)")
print("-" * 50)

# Create raw vector index
vdb = VectorDatabase()
raw_index = vdb.create(
    index_type="hnsw",
    dim=dim,
    space="cosine",
    m=16,
    ef_construction=200,
    expected_size=2000
)

print(f"📈 Index created: {raw_index.info()}")

# Add vectors
print("🔄 Adding vectors to raw index...")
add_result = raw_index.add(batch_data)
print(f"✅ Add result: {add_result.summary()}")

# Get stats before saving
stats_before = raw_index.get_stats()
print(f"📊 Index stats: {stats_before['total_vectors']} vectors, "
      f"storage: {stats_before['storage_mode_description']}")

# Save the raw index
print("💾 Saving raw index...")
try:
    raw_index.save("test_raw.zdb")
    print("✅ Raw index saved successfully!")
except Exception as e:
    print(f"❌ Raw index save failed: {e}")
    raise

print("\n🧪 TEST 2: Quantized Index (QuantizedWithRaw)")
print("-" * 50)

# Create quantized index with raw vector retention
quantization_config = {
    'type': 'pq',
    'subvectors': 8,
    'bits': 8,
    'training_size': 1000,
    'storage_mode': 'quantized_with_raw'
}

quantized_index = vdb.create(
    index_type="hnsw",
    dim=dim,
    space="cosine",
    quantization_config=quantization_config,
    expected_size=2000
)

print(f"📈 Quantized index created: {quantized_index.info()}")

# Add vectors (should trigger training)
print("🔄 Adding vectors to quantized index...")
add_result = quantized_index.add(batch_data)
print(f"✅ Add result: {add_result.summary()}")

# Check quantization status
quant_info = quantized_index.get_quantization_info()
if quant_info:
    print(f"🎯 Quantization trained: {quant_info['is_trained']}")
    print(f"   Compression ratio: {quant_info.get('compression_ratio', 'N/A')}")
    print(f"   Memory usage: {quant_info.get('memory_mb', 'N/A')} MB")

# Get stats
stats_quantized = quantized_index.get_stats()
print(f"📊 Quantized stats: {stats_quantized['total_vectors']} vectors, "
      f"storage: {stats_quantized['storage_mode_description']}")

# Save the quantized index
print("💾 Saving quantized index...")
try:
    quantized_index.save("test_quantized.zdb")
    print("✅ Quantized index saved successfully!")
except Exception as e:
    print(f"❌ Quantized index save failed: {e}")
    raise

print("\n🧪 TEST 3: Quantized Index (QuantizedOnly - Memory Optimized)")
print("-" * 50)

# Create memory-optimized quantized index
quantization_config_only = {
    'type': 'pq',
    'subvectors': 12,  # Higher compression
    'bits': 6,         # Fewer bits per centroid
    'training_size': 1000,
    'storage_mode': 'quantized_only'  # Don't keep raw vectors
}

quantized_only_index = vdb.create(
    index_type="hnsw",
    dim=dim,
    space="cosine",
    quantization_config=quantization_config_only,
    expected_size=2000
)

print(f"📈 Memory-optimized index created: {quantized_only_index.info()}")

# Add vectors
print("🔄 Adding vectors to memory-optimized index...")
add_result = quantized_only_index.add(batch_data)
print(f"✅ Add result: {add_result.summary()}")

# Check quantization status
quant_info_only = quantized_only_index.get_quantization_info()
if quant_info_only:
    print(f"🎯 Quantization trained: {quant_info_only['is_trained']}")
    print(f"   Compression ratio: {quant_info_only.get('compression_ratio', 'N/A')}")
    print(f"   Memory usage: {quant_info_only.get('memory_mb', 'N/A')} MB")

# Get stats
stats_only = quantized_only_index.get_stats()
print(f"📊 Memory-optimized stats: {stats_only['total_vectors']} vectors, "
      f"storage: {stats_only['storage_mode_description']}")

# Save the memory-optimized index
print("💾 Saving memory-optimized index...")
try:
    quantized_only_index.save("test_quantized_only.zdb")
    print("✅ Memory-optimized index saved successfully!")
except Exception as e:
    print(f"❌ Memory-optimized index save failed: {e}")
    raise

print("\n📁 VALIDATION: Checking Saved Files")
print("-" * 50)

# Validate saved directory structures
for test_dir in test_dirs:
    if os.path.exists(test_dir):
        print(f"\n📂 {test_dir}/")
        
        # List all files
        for file_path in sorted(Path(test_dir).glob("*")):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ├── {file_path.name} ({size_mb:.2f} MB)")
        
        # Read and display manifest
        manifest_path = Path(test_dir) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            print("   📋 Manifest Summary:")
            print(f"      ├── Format: {manifest['format_version']}")
            print(f"      ├── Vectors: {manifest['total_vectors']}")
            print(f"      ├── Storage: {manifest['storage_mode']}")
            print(f"      ├── Quantization: {'Yes' if manifest['has_quantization'] else 'No'}")
            if manifest.get('compression_info'):
                comp = manifest['compression_info']
                print(f"      ├── Compression: {comp['compression_ratio']:.1f}x")
            print(f"      └── Total Size: {manifest['total_size_mb']:.2f} MB")
        
        # Validate expected files are present
        expected_files = ['manifest.json', 'config.json', 'mappings.bin', 'metadata.json']
        missing_files = []
        for expected_file in expected_files:
            if not (Path(test_dir) / expected_file).exists():
                missing_files.append(expected_file)
        
        if missing_files:
            print(f"   ⚠️  Missing files: {missing_files}")
        else:
            print("   ✅ All core files present")

print("\n🎯 PHASE 1 TEST SUMMARY")
print("=" * 50)

# Final validation
all_passed = True
test_results = []

for test_dir in test_dirs:
    if os.path.exists(test_dir):
        # Check if manifest exists and is valid JSON
        manifest_path = Path(test_dir) / "manifest.json"
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            # Basic validation
            required_keys = ['format_version', 'total_vectors', 'storage_mode', 'files_included']
            missing_keys = [key for key in required_keys if key not in manifest]
            
            if missing_keys:
                test_results.append(f"❌ {test_dir}: Missing manifest keys: {missing_keys}")
                all_passed = False
            else:
                test_results.append(f"✅ {test_dir}: Valid manifest with {manifest['total_vectors']} vectors")
                
        except Exception as e:
            test_results.append(f"❌ {test_dir}: Manifest validation failed: {e}")
            all_passed = False
    else:
        test_results.append(f"❌ {test_dir}: Directory not created")
        all_passed = False

# Print results
for result in test_results:
    print(result)

if all_passed:
    print("\n🎉 ALL PHASE 1 TESTS PASSED!")
    print("✅ Save functionality working correctly")
    print("✅ All file formats valid")
    print("✅ Manifest generation successful")
    print("🚀 Ready for Phase 2 implementation!")
else:
    print("\n❌ SOME TESTS FAILED")
    print("Please check the errors above and fix issues before proceeding to Phase 2")

print("\n📝 Test completed. Saved directories available for inspection:")
for test_dir in test_dirs:
    if os.path.exists(test_dir):
        print(f"   - {test_dir}")

print("\n💡 Tip: Inspect the .zdb directories to see the file structure!")
print("💡 Tip: Check manifest.json files for detailed metadata!")