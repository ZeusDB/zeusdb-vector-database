#!/usr/bin/env python3
"""
Test Helper Functions for Load Implementation
Tests that our component loaders work correctly with existing saved data

RUN TEST #35 first to create the test data
"""

from zeusdb_vector_database import VectorDatabase
#import numpy as np
import os

print("🧪 Testing Helper Functions Implementation")
print("=" * 50)

# Use the existing test data from our previous successful save
test_dir = "test_phase2.zdb"

print(f"\n📁 Using existing test directory: {test_dir}")

# Verify the directory exists from our previous save test
if not os.path.exists(test_dir):
    print(f"❌ Test directory {test_dir} not found!")
    print("Please run the Phase 2 save test first to create the test data.")
    exit(1)

# Check what files we have to work with
files = os.listdir(test_dir)
print(f"📋 Available files: {sorted(files)}")

print("\n🔄 Testing Component Loading")
print("-" * 30)

# Test the helper functions by attempting to load
try:
    print("Attempting to call load method...")
    vdb = VectorDatabase()
    loaded_index = vdb.load(test_dir)
    print("✅ Load completed successfully!")
    
except Exception as e:
    print(f"Expected partial implementation result: {e}")
    
    # Check if we got the expected stopping point
    error_message = str(e)
    
    if "Basic component loading working" in error_message:
        print("\n✅ SUCCESS: Helper functions are working!")
        print("   - Config loading functional")
        print("   - Mappings loading functional") 
        print("   - Metadata loading functional")
        print("   - Vectors loading functional")
        print("   - Ready for next phase!")
        
    elif "Failed to read" in error_message:
        print(f"\n❌ File reading error: {error_message}")
        print("   - Check file permissions")
        print("   - Verify file format compatibility")
        
    elif "Failed to parse" in error_message:
        print(f"\n❌ File parsing error: {error_message}")
        print("   - Check serialization format compatibility")
        print("   - Verify data structure alignment")
        
    elif "not yet implemented" in error_message:
        print(f"\n⚠️  Helper functions not yet added: {error_message}")
        print("   - Need to add helper functions to persistence.rs")
        print("   - Need to rebuild with maturin develop")
        
    else:
        print(f"\n❓ Unexpected error: {error_message}")
        print("   - Review implementation")
        print("   - Check build status")

print("\n🎯 Expected Workflow:")
print("1. ✅ Load and parse manifest.json")
print("2. ✅ Load and parse config.json") 
print("3. ✅ Load and deserialize mappings.bin")
print("4. ✅ Load and parse metadata.json")
print("5. ✅ Load and deserialize vectors.bin")
print("6. 🔄 Stop with 'Basic component loading working' message")

print("\n🚀 Next Steps After Helper Functions Work:")
print("- Add HNSW graph loading functionality")
print("- Implement index reconstruction")
print("- Test complete save/load round-trip")
print("- Handle quantization scenarios")

print(f"\n📝 Test directory available: {test_dir}")
print("Ready to proceed with graph loading implementation!")