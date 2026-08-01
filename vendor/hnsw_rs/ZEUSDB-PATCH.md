# ZeusDB patch record for vendored hnsw_rs

Upstream crate: hnsw_rs 0.3.4 (crates.io, checksum
43a5258f079b97bf2e8311ff9579e903c899dcbac0d9a138d62e9a066778bd07).
This directory is a byte-for-byte copy of the registry source with one
deliberate change plus this file. Resolution is redirected here through
`[patch.crates-io]` in `vdb-core/Cargo.toml`.

## The change

File `src/hnsw.rs`, function `reverse_update_neighborhood_simple`.

```diff
-                    let l_n = n_to_add.point_ref.p_id.0 as usize;
+                    let l_n = l as usize;
```

## The defect

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

## On upgrade

This patch MUST be reapplied whenever the vendored copy is refreshed or
the version is bumped. The Rust regression test
`self_query_reachability` in `vdb-core/src/hnsw_index.rs` fails if the
patch is lost. Do not make any other change to this tree.
