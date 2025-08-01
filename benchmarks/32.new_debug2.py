from zeusdb_vector_database import VectorDatabase
vdb = VectorDatabase()
index = vdb.create('hnsw', dim=128)
print('✅ save method exists:', hasattr(index, 'save'))
print('🔍 Available methods:')
methods = [m for m in dir(index) if not m.startswith('_')]
for method in sorted(methods):
    if 'save' in method.lower() or 'load' in method.lower():
        print(f'   🎯 {method}')
    else:
        print(f'   - {method}')