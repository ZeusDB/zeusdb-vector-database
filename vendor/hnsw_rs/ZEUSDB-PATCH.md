# ZeusDB patch record for vendored hnsw_rs

Upstream crate: hnsw_rs 0.3.4 (crates.io, checksum
43a5258f079b97bf2e8311ff9579e903c899dcbac0d9a138d62e9a066778bd07).
This directory is a copy of the registry source with four deliberate
changes plus this file. Resolution is redirected here through
`[patch.crates-io]` in `vdb-core/Cargo.toml`.

## Patch 1. Reverse link layer assignment

File `src/hnsw.rs`, function `reverse_update_neighborhood_simple`.

```diff
-                    let l_n = n_to_add.point_ref.p_id.0 as usize;
+                    let l_n = l as usize;
```

### The defect

When a new point is inserted, the loop walks every layer `l` from the
point's top level down to zero and adds a reverse link from each of the
new point's neighbours back to the new point. Upstream files that link
at `n_to_add.point_ref.p_id.0`, the new point's own top level, on every
iteration instead of at the layer `l` being processed. For any point
whose top level is above zero, no reverse link is ever filed at layer
zero, so the point loses its layer-zero inbound adjacency and can become
unreachable to similarity search at any `ef_search`. The two expressions
coincide only on the first loop iteration, which is why points that stay
at level zero are unaffected.

## Patch 2. Deterministic level assignment

File `src/hnsw.rs`, `LayerGenerator::new`, `LayerGenerator::new_with_scale`,
and two added items.

```diff
-            rng: Arc::new(Mutex::new(StdRng::from_os_rng())),
+            rng: Arc::new(Mutex::new(StdRng::seed_from_u64(DEFAULT_LEVEL_SEED))),
```

applied at both construction sites, together with three additions.

- `pub const DEFAULT_LEVEL_SEED: u64 = 0x5A45_5553_4442_5F30;` declared
  above `LayerGenerator`.
- `LayerGenerator::set_seed`, which replaces the generator behind the
  existing `Arc<Mutex<StdRng>>`.
- `Hnsw::set_level_seed`, which forwards to it.

### Why it exists

Upstream seeds the layer assignment generator from OS entropy, so the
same data built with the same parameters produces a structurally
different graph on every run. An index is therefore not reproducible
from its inputs, and any A against B comparison is an unpaired
comparison across two different graphs. Measurement work on this crate
recorded unreachable point counts of 240, 245, 283, 447 and 658 on one
50,000 point dataset across five builds, a factor of 2.74 that is larger
than several of the effects being measured.

The generator drives level assignment only. `LayerGenerator::generate`
is the sole reader of the field, and `PointIndexation::generate_level`
is its sole caller, so a fixed seed changes nothing else. Level draws
stay independent of the data, so the level distribution is unchanged and
only the particular draw is now fixed.

`set_level_seed` resets the stream rather than extending it, so it must
be called before the first insert. Reloading a dumped index reconstructs
the generator through `new_with_scale`, which restarts the stream at the
default seed. Points appended to a reloaded index therefore repeat the
level sequence the original points drew. The draws remain independent of
the data, so this changes no distribution.

### What remains nondeterministic

`Hnsw::parallel_insert` reorders work by thread scheduling, so any path
that uses it still builds a different graph between runs. Seeding fixes
the sequential insert path only.

## Patch 3. Minimum inbound degree guard on the overflow pop

File `src/hnsw.rs`, function `reverse_update_neighborhood_simple`, with
supporting bookkeeping in `Hnsw::insert` and `Point`. File
`src/hnswio.rs`, function `HnswIo::load_point_indexation`.

```diff
-                            q_point_neighbours[l_n].pop();
+                            let list = &mut q_point_neighbours[l_n];
+                            let last = list.len() - 1;
+                            let mut victim = None;
+                            for i in (0..list.len()).rev() {
+                                if list[i].point_ref.in_degree[l_n].load(AtomOrd::Relaxed) >= 2 {
+                                    victim = Some(i);
+                                    break;
+                                }
+                            }
```

with the selected index removed instead of the last, falling back to the
last when no candidate qualifies, together with the supporting changes.

- `Point` gains `in_degree: [AtomicU32; NB_LAYER_MAX]`, the number of
  adjacency lists that name this point, one counter per layer.
- `Point` loses `#[derive(Clone)]`, because `AtomicU32` is not `Clone`,
  and gains an explicit `Clone` impl that snapshots the counters.
- Four sites that install an edge increment the counter for the layer
  they installed at. Three are in `Hnsw::insert` and
  `reverse_update_neighborhood_simple`, the fourth is in `hnswio.rs` so
  that a reloaded index does not start with every counter at zero.
- The one wholesale list replacement in `Hnsw::insert` decrements the
  targets it discards and increments the ones it installs, all under the
  write guard it already held.
- `GUARD_OVERFLOWS`, `GUARD_SAVES` and `GUARD_FALLBACKS` count the
  events, read through the added `guard_stats()`.

### Why it exists

The overflow pop is the only edge removal site in the crate. When a
reverse update pushes a neighbour list past its layer cap, upstream
discards the farthest entry unconditionally. If that entry was the only
adjacency list naming its target at that layer, the target is left with
no layer-zero in-edge. No search can then reach it through the graph,
whatever the `ef_search`, because graph traversal only ever arrives at a
point by following an in-edge. The guard walks from the farthest entry
inward and removes the first whose target would still hold at least one
inbound link afterwards, so the same number of edges is removed on every
overflow and only the choice of which changes.

Paired measurement on clustered data, 50 Gaussian clusters at sigma
0.15 in 768 dimensions, at 50,000 points across three seeds put the
recall difference at the shipped default at +0.0002, minus 0.0002 and
minus 0.0001, mixed in sign, with per-query paired p of 0.32, 0.41 and
0.56.
The matched-recall search width ratio is 1.005. Memory costs 51 bytes
per point and build time is unchanged. Unreachable points fall from 291,
261 and 210 to 27, 9 and 0, and no point is left with zero layer-zero
in-degree on any seed.

### Known limitation on the concurrent path

The guard reads a counter and then acts on it, and the two are not one
atomic step. The eviction site holds the write guard of the point whose
list is overflowing, but the counters it reads belong to the targets
named in that list, which are other points. Nothing in the crate makes a
thread take that lock before changing another point's counter, so the
read is unsynchronised with respect to every writer.

A read that is too low makes the guard decline to evict a candidate it
could safely have evicted. It falls through to the next candidate and at
worst to the farthest, which is the unpatched behaviour, so that
direction is a no-op.

A read that is too high is possible and is not a no-op. Two threads
overflowing two different lists that both name the same target can both
read its counter as 2, both conclude the target keeps a link, and both
evict, leaving it at zero. The pop also decrements after `Vec::remove`
rather than before, so between those two statements the counter reads
one higher than the true in-degree. In that case the guard fires wrongly
and strands a point, rather than failing to fire.

The effect is bounded. Exactly one edge is removed per overflow event,
the same as unpatched, so the guard never removes more edges than
upstream and strands at most one point per racing event. The counters do
not drift, because every list mutation is paired with exactly one atomic
read-modify-write, so once threads quiesce each counter equals the true
in-degree and a wrong eviction does not corrupt later decisions. A
stranding is repairable, since any later insert that selects the
stranded point installs an inbound link. A single thread cannot race
with itself, so only `Hnsw::parallel_insert` is exposed. In this
repository that is reached only from `insert_batch_pq`, and only when
the batch is at least 1000 times the rayon thread count, so the
quantized path can see it and the sequential default cannot.

This is recorded rather than fixed. Closing it needs either a compare
and swap on the counter or a lock ordering across points.

## Patch 4. The filtered search path

File `src/hnsw.rs`, function `search_layer`. Four changes in one place,
all of them consequences of a single decision.

```diff
-        return_points.push(Arc::new(PointWithOrder::new(
-            &entry_point,
-            dist_to_entry_point,
-        )));
+        let entry_admitted = match filter {
+            None => true,
+            Some(f) => f.hnsw_filter(&entry_point.get_origin_id()),
+        };
+        if entry_admitted {
+            return_points.push(Arc::new(PointWithOrder::new(
+                &entry_point,
+                dist_to_entry_point,
+            )));
+        }
```

The result heap is now seeded only with an entry point the filter admits.
Three further changes follow from it.

- The two reads of the farthest kept point, at the head of the candidate
  loop and inside the neighbour loop, become `match return_points.peek()`
  with `f32::INFINITY` for the empty case. An empty heap means nothing
  admissible has been found yet, so there is no stopping bound and the
  traversal continues.
- The stopping branch returns unconditionally. Upstream returned early
  only when `filter.is_none()` and otherwise ran an O(`ef`) `retain` over
  the heap on that turn and every turn after it. Every point in the heap
  now satisfies the filter, so the plain early return is correct with a
  filter as well as without one.
- The neighbour admission branch collapses to one filter test and one
  push, dropping the special case that cleared the heap when it held
  exactly one rejected element. That case can no longer arise.
- `let ef = ef.max(1);` is added beside `skiplist_size`. A width of zero
  leaves the heap no room, so under a filter it would empty itself after
  every admission, lose its stopping bound and walk the whole connected
  component. Without a filter the heap is never empty, so this changes
  nothing on the unfiltered path.

### Why it exists

ZeusDB passes a live-record predicate into every graph search, so that a
node stranded by `remove_point` or by `add(overwrite=True)` routes the
traversal but never consumes one of the `top_k` result slots. Upstream's
filtered path could not carry that load.

**It panicked.** `search_layer` seeded `return_points` with the entry
point whatever the filter said, then removed it again through the
`retain` on the stopping branch. If that emptied the heap and the popped
candidate had no unvisited neighbour, the loop took another turn and
`return_points.peek().unwrap()` unwrapped a `None`. Measured on a 4,000
point graph with half the ids filtered out, `search_filter` with `knbn`
of 0 and `ef_arg` of 0 panicked, and the same call with `filter` set to
`None` did not. ZeusDB reaches those arguments through
`search(vector, top_k=0, ef_search=0)`, since `search_filter` resolves
the width as `ef_arg.max(knbn)`. Reachable, and reachable only because a
filter is now passed.

**It returned nothing.** With the width resolving to 1 the single heap
slot was held by the rejected entry point, which `search_filter` then
dropped in its post-truncation filter, so the query returned an empty
list rather than the one live record it asked for. On the same graph,
`knbn` of 1 with `ef_arg` of 1 returned nothing on 25 of 300 queries at
half the ids filtered out, on 63 of 300 at nine tenths, and on 254 of 300
at 999 in 1,000. After the patch every one of those cells returns a
result on all 300 queries.

**It abandoned the early return.** With a filter present the stopping
condition retained instead of returning, so the search drained the whole
candidate heap and ran an O(`ef`) `retain` on every turn after the first.
That is a quadratic term on the hot path of every ZeusDB search.

## Total against the pristine registry copy

`src/hnsw.rs` differs by 249 lines, 189 added and 60 removed. Patch 1
accounts for 1 added and 1 removed. Patch 2 accounts for 30 added and 2
removed. Patch 3 accounts for 100 added and 3 removed. Patch 4 accounts
for 58 added and 54 removed, of which 20 of the additions are comment.
`src/hnswio.rs` differs by 4 lines, all added, all patch 3.
`ZEUSDB-PATCH.md` is an
added file. Five files carry no content change and differ only in line
endings, which relay 07 pinned across the repository:
`.cargo_vcs_info.json`, `.gitignore`, `Cargo.toml.orig`,
`LICENSE-APACHE` and `LICENSE-MIT`. `Cargo.toml` and `Cargo.lock` are
byte-identical to the registry copy. Nothing else in the tree differs.

## Detecting a lost patch

Patch 1 is caught by the Rust regression test `self_query_reachability`
in `vdb-core/src/hnsw_index.rs`, where roughly one to two percent of
self-queries fail without it.

Patch 2 cannot be caught by a test. An entropy seeded build simply
returns to passing or failing at random, so no assertion is stable
enough to depend on. Check for the absence of `from_os_rng` in this
file instead.

Patch 3 is caught by the Rust regression test `layer_zero_in_degree` in
`vdb-core/src/hnsw_index.rs`, which strands 24 of its 5,000 points
without it. That test builds at an `m` of 4 rather than the shipped 16
on purpose. The layer-zero neighbour cap is twice `m`, so a small `m`
fills lists early and makes the overflow pop frequent enough that the
test's uniform random vectors strand points at a size that runs in
seconds. Whether the shipped `m` of 16 strands points depends on the
data model rather than on index size. On uniform vectors it strands
none up to 30,000 points, while on clustered data, 50 Gaussian clusters
at sigma 0.15 in 768 dimensions, it strands 6 of 10,000 and the guard
strands none. The test counts in-degree from the adjacency lists rather
than from the counters this patch adds, so it holds against the graph
itself.

Patch 4 is caught partly by the Python regression tests
`test_search_after_deletes_returns_full_top_k` and
`test_repeatedly_overwritten_record_still_returns` in
`tests/test_compaction.py`, which exercise the filtered path and fail on the
returned count if it stops behaving. They do not distinguish this patch
from the call-site change in `vdb-core` that supplies the filter. The
panic and the empty result set need the boundary arguments, a `top_k` of
0 with an `ef_search` of 0 for the first and a `top_k` of 1 with an
`ef_search` of 1 for the second, and neither is in the suite because
both are degenerate parameter choices no ordinary caller makes. The
reliable check is textual. Without this patch `search_layer` contains
`return_points.retain(` and `let f = return_points.peek().unwrap();`.
With it, it contains neither.

## On upgrade

All four patches MUST be reapplied whenever the vendored copy is
refreshed or the version is bumped, and the line counts above rechecked
so a lost patch is visible. Both Rust regression tests insert
sequentially, so they depend on patch 2 for their own determinism. Patch
4 is load bearing rather than optional, because `vdb-core` passes a
filter into `search_filter` on every search call site and the unpatched
filtered path panics on some legal arguments. Do not make any other
change to this tree.
