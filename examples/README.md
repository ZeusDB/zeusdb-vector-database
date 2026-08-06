# Examples

Six runnable programs. Each one is standalone, needs no network and no embedding
provider, and finishes in a few seconds.

```bash
pip install zeusdb-vector-database numpy
python examples/01_search_a_document_collection.py
```

The package [README](../README.md) documents the API surface, parameter by
parameter. These files do something with it instead, and each one answers a
question the README states an answer to but does not show.

| File | Answers |
|------|---------|
| [01_search_a_document_collection.py](01_search_a_document_collection.py) | How do I get from a pile of vectors to a working search, and back again after a restart? Create, add, search, filter, save, load. |
| [02_metadata_filtering.py](02_metadata_filtering.py) | Why does my filter return nothing? Filtering runs after the search, so a selective filter over 4,000 products needs a `top_k` far larger than the page you want. Also covers the three ways a filter fails. |
| [03_quantization_tradeoff.py](03_quantization_tradeoff.py) | Should I turn quantization on, and which storage mode? Measures memory and recall for both modes with rerank on and off, on your machine. |
| [04_deletions_and_compaction.py](04_deletions_and_compaction.py) | My index takes deletions and updates. What is it accumulating, and what do I do about it? Runs a week of churn and calls `compact()`. |
| [05_concurrent_search.py](05_concurrent_search.py) | Can I serve queries from a thread pool, and can I write while I do? Eight threads searching, then an insert running underneath them. |
| [06_tuning_recall.py](06_tuning_recall.py) | How do I pick `m`, `ef_search` and `expected_size`? Measures recall across a grid of the first two, and shows what the third decides. |

## Output

Every file prints what it finds, and carries the full transcript at the foot of
the file in a string named `EXPECTED_OUTPUT`. Run one and you should see the
same thing.

A `...` in a transcript stands for a figure that moves between runs. That is
wall clock timing, and anything downstream of quantizer training, which trains
with an unseeded k-means and so does not repeat. Every one of those is printed
next to a word that does not move, such as `good` or `poor`, and the word is the
part worth reading.

`tests/test_examples.py` runs all six and checks each transcript, so an example
that stops working fails the suite rather than sitting here rotting.
