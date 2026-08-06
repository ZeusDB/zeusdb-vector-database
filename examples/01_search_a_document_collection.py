"""Build a searchable document collection, query it, save it, and load it back.

This is the whole arc in one file. Index a small collection, run a similarity
search, narrow it with a metadata filter, write the index to disk, and reopen it
in a state that answers the same questions.

Real applications get their vectors from an embedding model. This file builds
them by hand from six topic weights so that it runs offline in under a second,
and so that you can see for yourself why each result is close to the query.

Run it with:

    python 01_search_a_document_collection.py
"""

import os
import tempfile

from zeusdb_vector_database import VectorDatabase

# The six components of every vector in this file. A document's vector is how
# much of it is about each of these.
TOPICS = ["machine_learning", "cooking", "finance", "gardening", "astronomy", "travel"]

# id, title, topic weights, category, year
LIBRARY = [
    ("doc_01", "Training a classifier",       [0.9, 0.0, 0.1, 0.0, 0.0, 0.0], "tech",    2024),
    ("doc_02", "Neural nets from scratch",    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], "tech",    2025),
    ("doc_03", "Gradient descent explained",  [0.8, 0.0, 0.2, 0.0, 0.1, 0.0], "tech",    2023),
    ("doc_04", "Sourdough for beginners",     [0.0, 1.0, 0.0, 0.1, 0.0, 0.0], "food",    2024),
    ("doc_05", "Knife skills",                [0.0, 0.9, 0.0, 0.0, 0.0, 0.1], "food",    2022),
    ("doc_06", "Index funds explained",       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "finance", 2025),
    ("doc_07", "Reading a balance sheet",     [0.1, 0.0, 0.9, 0.0, 0.0, 0.0], "finance", 2023),
    ("doc_08", "Quantitative trading",        [0.7, 0.0, 0.7, 0.0, 0.0, 0.0], "finance", 2025),
    ("doc_09", "Pruning fruit trees",         [0.0, 0.1, 0.0, 1.0, 0.0, 0.0], "garden",  2024),
    ("doc_10", "Composting at home",          [0.0, 0.2, 0.0, 0.9, 0.0, 0.0], "garden",  2021),
    ("doc_11", "Photographing the moon",      [0.1, 0.0, 0.0, 0.0, 1.0, 0.2], "science", 2025),
    ("doc_12", "Planning an eclipse trip",    [0.0, 0.0, 0.1, 0.0, 0.8, 0.8], "science", 2024),
]


def build_index():
    """Create an index sized for the collection and load every document into it."""
    vdb = VectorDatabase()

    # dim must match the vectors exactly. expected_size is a declaration, not a
    # cap, and it is what picks the default graph degree, so declare it honestly.
    index = vdb.create(
        index_type="hnsw",
        dim=len(TOPICS),
        space="cosine",
        expected_size=len(LIBRARY),
    )

    # One add() call for the whole collection. Anything the index rejects is
    # reported back rather than raised, so check the result.
    result = index.add(
        [
            {
                "id": doc_id,
                "values": weights,
                "metadata": {"title": title, "category": category, "year": year},
            }
            for doc_id, title, weights, category, year in LIBRARY
        ]
    )
    print("indexed:", result.total_inserted, "documents,", result.total_errors, "errors")
    return index


def show(label, results):
    """Print a result page. Scores are distances, so lower is more similar."""
    print(label)
    for hit in results:
        print(f"  {hit['score']:.4f}  {hit['id']}  {hit['metadata']['title']}")


def main():
    index = build_index()
    print(index.info())
    print()

    # A query is a vector in the same space. This one asks for something mostly
    # about machine learning with a little finance in it.
    query = [0.8, 0.0, 0.4, 0.0, 0.0, 0.0]

    show("nearest 3 documents", index.search(vector=query, top_k=3))
    print()

    # Metadata filtering runs after the graph search, on the top_k the graph
    # returned. A selective filter therefore needs a larger top_k than the
    # number of results you want back, or it discards everything it was given.
    # The two garden documents come back at distance 1.0000, which under cosine
    # means they share no direction at all with the query.
    show(
        "filtered to garden, top_k=3 (nothing, no garden document is in the nearest 3)",
        index.search(vector=query, filter={"category": "garden"}, top_k=3),
    )
    print()
    show(
        "filtered to garden, top_k=12",
        index.search(vector=query, filter={"category": "garden"}, top_k=12),
    )
    print()

    # Index level metadata is a flat str to str map that travels with the index.
    # It is separate from the per-record metadata used for filtering.
    index.add_metadata({"collection": "demo_library", "topics": ",".join(TOPICS)})

    with tempfile.TemporaryDirectory() as workspace:
        path = os.path.join(workspace, "library.zdb")

        # save() writes a directory, and prints a progress banner to stdout that
        # is not part of the logging system and cannot be turned off.
        index.save(path)
        print("saved files:", sorted(os.listdir(path)))
        print()

        # load() rebuilds the graph from the stored records, so it costs roughly
        # what the original insert cost rather than what the directory weighs.
        reopened = VectorDatabase().load(path)

    print("reopened:", reopened.get_vector_count(), "documents")
    print("collection:", reopened.get_metadata("collection"))
    show("the same query against the reopened index", reopened.search(vector=query, top_k=3))
    print()

    # Under cosine the index normalises vectors on the way in, so a vector read
    # back is the unit length form and not the numbers that were supplied.
    stored = reopened.get_records("doc_08")[0]
    print("doc_08 as supplied:", LIBRARY[7][2])
    print("doc_08 as stored:  ", [round(v, 4) for v in stored["vector"]])


# The transcript this file prints, with the progress banner save() and load()
# write to stdout left out.
EXPECTED_OUTPUT = """\
indexed: 12 documents, 0 errors
HNSWIndex(dim=6, space=cosine, m=16, ef_construction=200, expected_size=12, vectors=12, quantization=none)

nearest 3 documents
  0.0309  doc_03  Gradient descent explained
  0.0513  doc_08  Quantitative trading
  0.0617  doc_01  Training a classifier

filtered to garden, top_k=3 (nothing, no garden document is in the nearest 3)

filtered to garden, top_k=12
  1.0000  doc_09  Pruning fruit trees
  1.0000  doc_10  Composting at home

saved files: ['config.json', 'hnsw_index.hnsw.data', 'hnsw_index.hnsw.graph', 'manifest.json', 'mappings.bin', 'metadata.json', 'vectors.bin']

reopened: 12 documents
collection: demo_library
the same query against the reopened index
  0.0309  doc_03  Gradient descent explained
  0.0513  doc_08  Quantitative trading
  0.0617  doc_01  Training a classifier

doc_08 as supplied: [0.7, 0.0, 0.7, 0.0, 0.0, 0.0]
doc_08 as stored:   [0.7071, 0.0, 0.7071, 0.0, 0.0, 0.0]
"""

if __name__ == "__main__":
    main()
