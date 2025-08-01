#!/usr/bin/env python3
"""
Simple script to test if the debug_test method is picked up
"""

def test_debug_method():
    """Test if debug_test method is available"""
    
    print("🧪 Testing debug_test method detection")
    print("=" * 40)
    
    try:
        # Step 1: Import
        print("🔄 Step 1: Importing VectorDatabase...")
        from zeusdb_vector_database import VectorDatabase
        print("✅ Import successful")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    try:
        # Step 2: Create instance
        print("🔄 Step 2: Creating VectorDatabase instance...")
        vdb = VectorDatabase()
        print("✅ VectorDatabase created")
        
    except Exception as e:
        print(f"❌ VectorDatabase creation failed: {e}")
        return False
    
    try:
        # Step 3: Create index
        print("🔄 Step 3: Creating HNSW index...")
        index = vdb.create("hnsw", dim=128)
        print("✅ HNSW index created")
        print(f"   Index type: {type(index)}")
        
    except Exception as e:
        print(f"❌ HNSW index creation failed: {e}")
        return False
    
    # Step 4: List all available methods
    print("🔄 Step 4: Listing all available methods...")
    all_methods = [method for method in dir(index) if not method.startswith('_')]
    print(f"✅ Found {len(all_methods)} methods:")
    for method in sorted(all_methods):
        print(f"   - {method}")
    
    # Step 5: Check specifically for debug_test
    print("🔄 Step 5: Checking for debug_test method...")
    if hasattr(index, 'debug_test'):
        print("✅ debug_test method FOUND!")
        
        # Step 6: Try to call it
        print("🔄 Step 6: Calling debug_test method...")
        try:
            result = index.debug_test()
            print(f"✅ debug_test() returned: '{result}'")
            
            if result == "debug works":
                print("🎉 SUCCESS! New methods ARE being picked up!")
                return True
            else:
                print(f"⚠️  Unexpected result: expected 'debug works', got '{result}'")
                return True  # Still counts as success
                
        except Exception as e:
            print(f"❌ debug_test() call failed: {e}")
            return False
            
    else:
        print("❌ debug_test method NOT FOUND")
        print("   This means new methods are NOT being picked up")
        print("   Build system issue detected!")
        return False

def show_diagnosis():
    """Show what the results mean"""
    print("\n" + "=" * 50)
    print("🔍 DIAGNOSIS")
    print("=" * 50)
    
    success = test_debug_method()
    
    if success:
        print("\n✅ BUILD SYSTEM IS WORKING!")
        print("   - New methods are being picked up")
        print("   - Your save method should work too")
        print("   - The issue might be elsewhere")
        
        print("\n🔧 Next steps:")
        print("   1. Check if 'save' method exists: hasattr(index, 'save')")
        print("   2. If save exists, try calling it")
        print("   3. If save missing, check it's in the same #[pymethods] block")
        
    else:
        print("\n❌ BUILD SYSTEM ISSUE DETECTED!")
        print("   - New methods are NOT being picked up")
        print("   - This explains why 'save' method isn't available")
        
        print("\n🔧 Next steps:")
        print("   1. Check Cargo.toml has: crate-type = ['cdylib']")
        print("   2. Verify lib.rs has: mod persistence;")
        print("   3. Try complete clean rebuild:")
        print("      pip uninstall zeusdb-vector-database -y")
        print("      cargo clean")
        print("      maturin develop --release --force")

if __name__ == "__main__":
    show_diagnosis()