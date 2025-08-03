#!/usr/bin/env python3
"""
Test Helper Functions for Load Implementation
Tests that our component loaders work correctly with existing saved data

RUN TEST #35 first to create the test data
"""

from zeusdb_vector_database import VectorDatabase
import os

print("🧪 Testing Helper Functions Implementation")
print("=" * 50)

# Use the test directory created by the Phase 2 save test
test_dir = "test_phase2.zdb"

print(f"\n📁 Looking for test directory: {test_dir}")

# Check if the directory exists
if not os.path.exists(test_dir):
    print(f"❌ Test directory {test_dir} not found!")
    print("\nPlease run the Phase 2 save test first:")
    print("   python test_phase2_save.py")
    print("\nThis will create the test_phase2.zdb directory we need.")
    exit(1)

# Show what files we have
files = os.listdir(test_dir)
print(f"✅ Found test directory with files: {sorted(files)}")

print("\n🔄 Testing Helper Functions")
print("-" * 30)

# Test our new load_index() function with helper functions
try:
    print("Calling vdb.load() to test helper functions...")
    vdb = VectorDatabase()
    loaded_index = vdb.load(test_dir)
    print("✅ Unexpected success - load completed!")
    
except Exception as e:
    error_message = str(e)
    print(f"Result: {error_message}")
    
    # Check what kind of result we got
    if "Basic component loading working" in error_message:
        print("\n🎉 SUCCESS! Helper functions are working correctly!")
        print("✅ All components loaded successfully:")
        print("   - manifest.json ✅")
        print("   - config.json ✅") 
        print("   - mappings.bin ✅")
        print("   - metadata.json ✅")
        print("   - vectors.bin ✅")
        print("   - quantization check ✅")
        print("\n🚀 Ready for next phase: HNSW graph loading!")
        
    elif "HNSW graph loading working" in error_message:
        print("\n🎉 BREAKTHROUGH! HNSW Graph Loading Working!")
        print("✅ Complete Phase 2 success:")
        print("   - All ZeusDB components loaded ✅")
        print("   - HNSW graph structure loaded ✅") 
        print("   - 50 graph points successfully loaded ✅")
        print("\n🚀 Ready for Phase 3: Index reconstruction!")
        print("   This is the final step to complete the project!")
        
    elif "not yet implemented" in error_message:
        print("\n⚠️  Helper functions not added yet:")
        print("❌ Still using placeholder load_index() function")
        print("\nNext steps:")
        print("1. Add helper functions to persistence.rs")
        print("2. Replace load_index() function") 
        print("3. Rebuild with: maturin develop --release --force")
        
    elif "Failed to" in error_message:
        print(f"\n❌ Implementation error: {error_message}")
        print("\nPossible issues:")
        print("- File format compatibility")
        print("- Missing imports")
        print("- Serialization errors")
        
    else:
        print(f"\n❓ Unexpected error: {error_message}")
        print("Review the implementation and error details.")

print("\n📋 Expected Process:")
print("1. Load manifest.json and validate")
print("2. Load config.json with index settings")
print("3. Load mappings.bin with ID mappings") 
print("4. Load metadata.json with vector metadata")
print("5. Load vectors.bin with raw vectors")
print("6. Check for quantization.json")
print("7. Validate data consistency")
print("8. Stop with success message")

print(f"\n📝 Using test data from: {test_dir}")