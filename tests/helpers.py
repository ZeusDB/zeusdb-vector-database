"""Shared assertion helpers for the vector database tests."""


def normalize_vector(vector):
    """Normalize vector for cosine distance (same as Rust implementation)"""
    import math
    norm = math.sqrt(sum(x * x for x in vector))
    if norm > 0.0:
        return [x / norm for x in vector]
    return vector

def assert_vectors_close(actual, expected, tolerance=1e-6, space="cosine"):
    """Assert vectors are close, accounting for normalization"""
    if space.lower() == "cosine":
        expected = normalize_vector(expected)
    
    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert abs(a - e) < tolerance, f"Vector element {i}: expected {e}, got {a}"
