# from zeusdb_vector_database import VectorDatabase
# print(dir(VectorDatabase))


from zeusdb_vector_database import VectorDatabase

# Create the VectorDatabase and then create an index
vdb = VectorDatabase()
index = vdb.create("hnsw", dim=128)

print("🔍 HNSWIndex type and methods:")
print(f"Type: {type(index)}")
print(f"Class: {index.__class__}")

print("\n📋 All methods on HNSWIndex:")
methods = dir(index)
for method in sorted(methods):
    if not method.startswith('__'):  # Skip dunder methods
        print(f"   - {method}")

print(f"\n❓ Has 'save' method: {hasattr(index, 'save')}")

# Try to call a method we know exists
try:
    print(f"\n✅ Can call get_stats(): {type(index.get_stats())}")
except Exception as e:
    print(f"❌ Error calling get_stats(): {e}")

# Check if we can see any persistence-related methods
persistence_methods = [m for m in dir(index) if 'save' in m.lower() or 'load' in m.lower() or 'persist' in m.lower()]
print(f"\n💾 Persistence-related methods: {persistence_methods}")
