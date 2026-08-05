"""Run a churning index, watch the graph fill with debris, and reclaim it.

An index that is written once and read forever needs nothing from this file. An
index that takes deletions and updates does, because neither one removes the
node it leaves in the HNSW graph. Searches never return those nodes, but they
hold memory and edge slots, and nothing clears them automatically.

This file runs a week of churn against 2,000 records, shows the debris
accumulating in `get_stats()`, calls `compact()`, and checks that every record
survived it unchanged.

Run it with:

    python 04_deletions_and_compaction.py
"""

import numpy as np

from zeusdb_vector_database import VectorDatabase

DIM = 32
RECORDS = 2000
DAYS = 7


def report(label, index):
    """graph_nodes minus stranded_graph_nodes is what search actually walks."""
    stats = index.get_stats()
    print(
        f"  {label:<22s} records {stats['total_vectors']:>5s}   "
        f"graph nodes {stats['graph_nodes']:>5s}   "
        f"stranded {stats['stranded_graph_nodes']:>5s}"
    )


def main():
    rng = np.random.default_rng(4242)
    vectors = rng.standard_normal((RECORDS, DIM))
    vectors = (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(np.float32)
    ids = [f"rec_{i:04d}" for i in range(RECORDS)]

    index = VectorDatabase().create("hnsw", dim=DIM, expected_size=RECORDS)
    index.add(
        {
            "ids": ids,
            "embeddings": vectors,
            "metadatas": [{"day": 0, "state": "new"} for _ in ids],
        }
    )
    print("a week of churn")
    report("after the initial load", index)

    # ------------------------------------------------------------------
    # The churn
    # ------------------------------------------------------------------
    # Each day removes 40 records and updates another 80. remove_point() is a
    # logical delete, and add() on an existing id is an upsert, so both leave the
    # old graph node behind.
    live = set(ids)
    for day in range(1, DAYS + 1):
        doomed = ids[(day - 1) * 40 : day * 40]
        for record_id in doomed:
            index.remove_point(record_id)
        live -= set(doomed)

        refreshed = ids[1000 + (day - 1) * 80 : 1000 + day * 80]
        index.add(
            {
                "ids": refreshed,
                "embeddings": vectors[[ids.index(r) for r in refreshed]],
                "metadatas": [{"day": day, "state": "updated"} for _ in refreshed],
            }
        )
        report(f"end of day {day}", index)

    print()
    print("The record count falls and the graph does not. Every removal and every")
    print("update adds one stranded node, and searches walk past all of them.")
    print()

    # ------------------------------------------------------------------
    # Reclaiming
    # ------------------------------------------------------------------
    # Take a reference before compacting so the check afterwards is real.
    query = vectors[1500]
    before_hits = [(h["id"], round(h["score"], 6)) for h in index.search(query, top_k=5)]
    before_count = index.get_vector_count()

    # compact() rebuilds the graph in memory and returns the number of nodes it
    # reclaimed. It costs a full rebuild, proportional to the live records rather
    # than to the amount of debris, and it holds both graphs while it runs.
    reclaimed = index.compact()
    print(f"compact() reclaimed {reclaimed} nodes")
    report("after compact", index)
    print(f"compact() again reclaimed {index.compact()} nodes, there is nothing left")
    print()

    after_hits = [(h["id"], round(h["score"], 6)) for h in index.search(query, top_k=5)]
    print("records before and after:", before_count, index.get_vector_count())
    print("the same query returns the same page:", before_hits == after_hits)
    print("a removed id is still absent:", not index.contains(ids[0]))
    print("a live id still resolves:", index.contains(ids[1999]))
    print()

    # ------------------------------------------------------------------
    # An update replaces metadata wholesale
    # ------------------------------------------------------------------
    # This catches people out. add() on an existing id removes the record first
    # and inserts the new one, so a key left out of the new metadata is gone
    # rather than kept from the old record.
    # Metadata comes back as a dict whose key order is not stable between runs,
    # so anything that prints one whole should sort it first.
    def metadata_of(record_id):
        record = index.get_records(record_id, return_vector=False)[0]
        return dict(sorted(record["metadata"].items()))

    index.add({"id": "rec_1999", "values": vectors[1999], "metadata": {"day": 9, "owner": "ops"}})
    print("metadata after a full update:", metadata_of("rec_1999"))
    index.add({"id": "rec_1999", "values": vectors[1999], "metadata": {"day": 10}})
    print("owner was not carried over:  ", metadata_of("rec_1999"))

    # overwrite=False refuses the collision instead. It does not raise. The
    # record is skipped and counted in the AddResult.
    refused = index.add(
        {"id": "rec_1999", "values": vectors[1999], "metadata": {"day": 11}}, overwrite=False
    )
    print("with overwrite=False:", refused.total_inserted, "inserted,", refused.total_errors, "error")
    print("  ", refused.errors[0])


# The transcript this file prints.
EXPECTED_OUTPUT = """\
a week of churn
  after the initial load records  2000   graph nodes  2000   stranded     0
  end of day 1           records  1960   graph nodes  2080   stranded   120
  end of day 2           records  1920   graph nodes  2160   stranded   240
  end of day 3           records  1880   graph nodes  2240   stranded   360
  end of day 4           records  1840   graph nodes  2320   stranded   480
  end of day 5           records  1800   graph nodes  2400   stranded   600
  end of day 6           records  1760   graph nodes  2480   stranded   720
  end of day 7           records  1720   graph nodes  2560   stranded   840

The record count falls and the graph does not. Every removal and every
update adds one stranded node, and searches walk past all of them.

compact() reclaimed 840 nodes
  after compact          records  1720   graph nodes  1720   stranded     0
compact() again reclaimed 0 nodes, there is nothing left

records before and after: 1720 1720
the same query returns the same page: True
a removed id is still absent: True
a live id still resolves: True

metadata after a full update: {'day': 9, 'owner': 'ops'}
owner was not carried over:   {'day': 10}
with overwrite=False: 0 inserted, 1 error
   Vector rec_1999: ValueError: Vector with ID 'rec_1999' already exists
"""

if __name__ == "__main__":
    main()
