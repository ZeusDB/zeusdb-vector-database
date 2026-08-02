# ZeusDB patch record for vendored hnsw_rs

Upstream crate: hnsw_rs 0.3.4 (crates.io, checksum
43a5258f079b97bf2e8311ff9579e903c899dcbac0d9a138d62e9a066778bd07).
This directory is a copy of the registry source with two deliberate
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

## Total against the pristine registry copy

`src/hnsw.rs` differs by 34 lines, 31 added and 3 removed. Patch 1
accounts for 1 added and 1 removed. Patch 2 accounts for 30 added and 2
removed. `ZEUSDB-PATCH.md` is an added file. Five files carry no content
change and differ only in line endings, which relay 07 pinned across the
repository: `.cargo_vcs_info.json`, `.gitignore`, `Cargo.toml.orig`,
`LICENSE-APACHE` and `LICENSE-MIT`. Nothing else in the tree differs.

## On upgrade

Both patches MUST be reapplied whenever the vendored copy is refreshed
or the version is bumped, and the line count above rechecked so a lost
patch is visible. The Rust regression test `self_query_reachability` in
`vdb-core/src/hnsw_index.rs` fails if patch 1 is lost. It inserts
sequentially so that it depends on patch 2 for its own determinism, but
it cannot detect the loss of patch 2, since an entropy seeded build
would simply return to passing or failing at random. Check for
`from_os_rng` in this file instead. Do not make any other change to this
tree.
